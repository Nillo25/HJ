"""Скан -> граф сцены.

    # моделью YOLO-seg
    python -m fp.infer --image plan.png --model runs/segment/train/weights/best.pt --out scenes/plan
    # из разметки CubiCasa (без модели — для демо чата и как эталон)
    python -m fp.infer --image F1_scaled.png --gt-svg model.svg --out scenes/demo
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES
from .cubicasa_svg import parse_cubicasa_svg
from .preprocess import preprocess
from .scene import Scene

# допуск упрощения полигона (доля периметра): стены/комнаты — прямые углы, иконки — грубее
EPS = {"wall": 0.004, "door": 0.02, "window": 0.02}
EPS_ROOM, EPS_ICON = 0.006, 0.02


def simplify(poly: np.ndarray, cls: str) -> list:
    cnt = poly.reshape(-1, 1, 2).astype(np.float32)
    eps = EPS.get(cls, EPS_ROOM if cls.startswith("room_") else EPS_ICON)
    approx = cv2.approxPolyDP(cnt, eps * cv2.arcLength(cnt, True), True)
    return approx.reshape(-1, 2).tolist() if len(approx) >= 3 else poly.tolist()


THRESHOLDS = json.loads((Path(__file__).parent / "thresholds.json").read_text())


def _ios(a, b) -> float:
    """Площадь пересечения боксов / площадь меньшего."""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-6)


def merge_duplicates(items: list, thr: float) -> list:
    """items = [(poly, conf, box)]; склеивает группы с пересечением/меньший >= thr
    (одно окно, разбитое моделью на две детекции). Полигон группы — контур объединения."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _ios(items[i][2], items[j][2]) >= thr:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(items[i])
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        u = unary_union([Polygon(p).buffer(0) for p, _, _ in g if len(p) >= 3])
        geom = max(u.geoms, key=lambda x: x.area) if u.geom_type == "MultiPolygon" else u
        poly = np.asarray(geom.convex_hull.exterior.coords)[:-1] if u.geom_type == "MultiPolygon" \
            else np.asarray(geom.exterior.coords)[:-1]
        box = [min(b[0] for _, _, b in g), min(b[1] for _, _, b in g), max(b[2] for _, _, b in g), max(b[3] for _, _, b in g)]
        out.append((poly, max(c for _, c, _ in g), box))
    return out


def detect_yolo(img: np.ndarray, model_path: str, imgsz: int = 1024, thresholds: dict | None = None,
                candidates: list | None = None):
    """candidates (если передан) пополняется пограничными детекциями, НЕ прошедшими порог:
    [cls, conf, cx, cy] — по ним query считает уверенность ответа (см. thresholds.json, _calibration)."""
    from ultralytics import YOLO  # тяжёлый импорт — только когда нужен

    th = thresholds or THRESHOLDS
    model = _model_cache.get(model_path) or _model_cache.setdefault(model_path, YOLO(model_path))
    margin = th.get("_calibration", {}).get("near_margin", 0.15)
    min_conf = max(0.01, min(v["conf"] for k, v in th.items() if not k.startswith("_")) - margin)
    # max_det=1000: на планах бывает до ~600 объектов (стены дробятся), дефолт 300 срезает
    res = model.predict(img, imgsz=imgsz, conf=min_conf, retina_masks=True, max_det=1000, verbose=False)[0]
    if res.masks is None:
        return []
    by_cls: dict = {}
    for poly, c, p, box in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
        cls = res.names[int(c)]
        if cls not in CLASSES or len(poly) < 3:
            continue
        # порог по классу (подобран под точный подсчёт, scripts/tune_thresholds.py)
        t = th.get(cls, th["default"])["conf"]
        if p < t:
            if candidates is not None and p >= t - margin:
                candidates.append([cls, round(float(p), 3), round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)])
            continue
        by_cls.setdefault(cls, []).append((np.asarray(poly), float(p), box))
    dets = []
    for cls, items in by_cls.items():
        merge = th.get(cls, th["default"])["merge"]
        if merge:
            items = merge_duplicates(items, merge)
        # NB: conf = уверенность детекции YOLO (в confidence.md — средняя вероятность по маске;
        # у Ultralytics маски бинаризуются, поэтому берём conf бокса)
        dets += [(cls, simplify(poly, cls), conf) for poly, conf, _ in items]
    return dets


_model_cache: dict = {}


def detect_gt(svg_path: Path, w: int, h: int, image: Path | None = None):
    from .convert_yolo import svg_to_image_scale
    parsed = parse_cubicasa_svg(svg_path)
    sx, sy = svg_to_image_scale(svg_path.parent, image, w, h) if image else (1.0, 1.0)
    return [(c, [(x * sx, y * sy) for x, y in poly], 1.0) for c, poly in parsed.objects]


def run(image: Path, out: Path, model: str | None = None, gt_svg: Path | None = None,
        do_preprocess: bool = True, auto_scale: bool | None = None) -> Scene:
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"не читается картинка: {image}")
    if gt_svg:
        # координаты разметки привязаны к исходной картинке — не трансформируем
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        info = {"preprocess": False}
        h, w = img.shape[:2]
        dets, source = detect_gt(gt_svg, w, h, image), "gt"
    else:
        if not model:
            raise ValueError("нужен --model (веса YOLO-seg) или --gt-svg")
        if do_preprocess:
            img, info = preprocess(img)
        else:
            info = {"preprocess": False}
        h, w = img.shape[:2]
        cands: list = []
        dets, source = detect_yolo(img, model, candidates=cands), "yolo"
        info = {**info, "candidates": cands}
    cv2.imwrite(str(out / "image.png"), img)
    sc = Scene.build("image.png", w, h, dets, source=source,
                     meta={"source_image": image.name, "model": model, **info})
    # масштаб: один вызов VLM на план (подписи площадей / линейная шкала), см. fp/scale.py
    if auto_scale is None:
        auto_scale = os.environ.get("FP_AUTO_SCALE", "1") != "0"
    if auto_scale:
        analyze_sheet(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), sc)
    sc.save(out)
    return sc


def analyze_sheet(img: np.ndarray, sc: Scene) -> Scene:
    """Один вызов VLM на лист: штамп, легенда, таблицы, примечания, шкала, подписи площадей.
    Затем без VLM: масштаб (fp.scale) и зоны по цветам легенды (fp.sheet.find_zones) — в сцену."""
    from .scale import estimate
    from .scene import Element
    from .sheet import find_zones, read_sheet
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sc.meta["scale"] = {"px_per_m": None, "source": None, "detail": "нет ключа для VLM — задайте масштаб вручную"}
        return sc
    try:
        sheet, cost = read_sheet(img)
    except Exception as e:  # noqa: BLE001
        sc.meta["scale"] = {"px_per_m": None, "source": None, "detail": f"VLM не ответила: {e}"}
        return sc
    sc.meta["sheet"] = sheet
    est = estimate(img, sc, read=sheet, cost=cost)
    # шкалу не нашли, а масштаб есть только из текста — один узкий вызов: поиск шкалы по четвертям листа
    if not sheet.get("scale_bar") and est.get("source") in (None, "scale_text"):
        from .sheet import find_scale_bar
        try:
            bar, c2 = find_scale_bar(img)
        except Exception:  # noqa: BLE001
            bar, c2 = None, 0.0
        cost += c2
        if bar:
            sheet["scale_bar"] = bar
            est = estimate(img, sc, read=sheet, cost=cost)
    sc.meta["sheet_cost_usd"] = round(cost, 6)
    sc.meta["scale"] = est
    sc.meta["px_per_m"] = est.get("px_per_m")
    # зоны легенды: убираем старые, ищем цвета на плане (кроме самих образцов легенды)
    sc.objects = [o for o in sc.objects if o.cls != "zone"]
    legend = sheet.get("legend", [])
    zones = find_zones(img, legend, region=sheet.get("drawing_bbox"),
                       exclude=[lg["swatch_bbox"] for lg in legend if lg.get("swatch_bbox")])
    sc.meta["zones"] = {}
    for n, z in enumerate(zones, 1):
        zid = f"zone{n}"
        sc.objects.append(Element(zid, "zone", z["polygon"], 0.8, source="legend"))
        sc.meta["zones"][zid] = {"label": z["label"], "color": z["color"]}
    # помещения: площади и плинтусы (подписи + заливка по стенам + контуры модели)
    from .spaces import build as build_spaces
    build_spaces(img, sc)
    # второе мнение ИИ по сантехнике — только для уверенности, детекции не меняет (fp/arbiter.py)
    from .arbiter import check_fixtures
    sc.meta["fixture_check"] = check_fixtures(img, sc)
    sc.meta["sheet_cost_usd"] = round(sc.meta.get("sheet_cost_usd", 0) + sc.meta["fixture_check"].get("cost_usd", 0), 6)
    return sc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model")
    ap.add_argument("--gt-svg", type=Path)
    ap.add_argument("--no-preprocess", action="store_true")
    a = ap.parse_args(argv)
    sc = run(a.image, a.out, a.model, a.gt_svg, not a.no_preprocess)
    print(f"объектов: {len(sc.objects)}, комнат: {len(sc.rooms)} -> {a.out}")


if __name__ == "__main__":
    main()
