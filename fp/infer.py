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
import re
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


ARC_MIN_PPM = 33.0  # двери по дугам — только если дверь 0.6 м на листе не меньше 20 px
WALL_ROOMS_MIN_PPM = 20.0  # помещения по стенам (рамки от ИИ) — только если комната 3 м на листе ≥ 60 px

# менять при правках разбора, влияющих на сцену: повторная загрузка того же файла тогда разберёт его заново
PIPELINE_VERSION = "2026-11-14.scale4+sheet-tiles-door-verify"

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
        original = img  # полное разрешение — для тайлов и проверки дверей на чертёжных листах (fp/tiles.py)
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
        full = f = None
        if not gt_svg:
            from .tiles import full_res
            full = full_res(original, info)
            f = w / full.shape[1]
        analyze_sheet(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), sc, full=full, f=f)
    sc.save(out)
    return sc


def drop_outside_drawing(sc: Scene, bbox, pad: float = 0.02, max_frac: float | None = 0.3) -> int:
    """Убрать детекции вне основного чертежа: штамп, легенда, таблицы, ключевой план. Модель обучена на квартирах
    и на чертёжных листах принимает графы штампа за комнаты. bbox — рамка чертежа от VLM (fp.sheet)."""
    if not bbox or len(bbox) != 4:
        return 0
    x0, y0, x1, y1 = bbox
    px, py = pad * sc.width, pad * sc.height
    x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py

    def inside(e):
        c = e.shape().centroid
        return x0 <= c.x <= x1 and y0 <= c.y <= y1
    before = len(sc.objects) + len(sc.rooms)
    keep_rooms = [r for r in sc.rooms if inside(r)]
    keep_objs = [o for o in sc.objects if o.source == "legend" or inside(o)]
    removed = before - len(keep_objs) - len(keep_rooms)
    # рамка от VLM бывает неполной (на листе два этажа, а рамка вокруг одного) — тогда не трогаем
    if max_frac is not None and before and removed > max_frac * before:
        sc.meta["outside_drawing_skipped"] = f"рамка чертежа убрала бы {removed} из {before} — не применена"
        return 0
    sc.rooms, sc.objects = keep_rooms, keep_objs
    if removed:
        sc.meta["outside_drawing_removed"] = removed
        sc.assign_rooms()
    return removed


def analyze_sheet(img: np.ndarray, sc: Scene, full: np.ndarray | None = None, f: float | None = None) -> Scene:
    """Один вызов VLM на лист: штамп, легенда, таблицы, примечания, шкала, подписи площадей.
    Затем без VLM: масштаб (fp.scale) и зоны по цветам легенды (fp.sheet.find_zones) — в сцену.
    full — лист в полном разрешении (после того же поворота/обрезки), f — full -> координаты сцены;
    без него тайлы и проверка дверей работают по img (при повторном анализе сохранённой сцены)."""
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
    tb = sheet.get("title_block") or {}
    if tb.get("sheet_number"):  # чертёжный лист со штампом и номером — рамке чертежа доверяем
        drop_outside_drawing(sc, sheet.get("drawing_bbox"), max_frac=None)
    elif sheet.get("legend") or sheet.get("tables"):  # есть легенда/таблицы, но не уверены, что это целый лист
        drop_outside_drawing(sc, sheet.get("drawing_bbox"))
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
    # «комнаты» меньше 0.6 м² (кроме санузлов) — шум модели (чертёжные листы, штриховка); убираем, когда масштаб известен
    if est.get("px_per_m"):
        small = [r.id for r in sc.rooms
                 if r.cls != "room_bath" and r.shape().area / est["px_per_m"] ** 2 < 0.6]
        if small:
            sc.rooms = [r for r in sc.rooms if r.id not in small]
            sc.meta["tiny_rooms_removed"] = small
            sc.assign_rooms()
    # почти прямоугольные контуры модели — по линиям стен: маска у перегородок с дверями «гуляет»
    # (заходит в соседнее помещение, даёт ступеньки), а от контура зависят площадь и плинтус
    if est.get("px_per_m"):
        from .rooms_cv import rectify_room
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        fixed = []
        for r in sc.rooms:
            if r.source != "yolo":
                continue
            new = rectify_room(gray, r.polygon, est["px_per_m"])
            if new:
                r.polygon = [[float(x), float(y)] for x, y in new]
                fixed.append(r.id)
        sc.meta["rooms_rectified"] = fixed
        if fixed:
            sc.assign_rooms()
    # зоны легенды: убираем старые, ищем цвета на плане (кроме самих образцов легенды)
    sc.objects = [o for o in sc.objects if o.cls != "zone"]
    legend = sheet.get("legend", [])
    zones = find_zones(img, legend, region=sheet.get("drawing_bbox"),
                       exclude=[lg["swatch_bbox"] for lg in legend if lg.get("swatch_bbox")])
    sc.meta["zones"] = {}
    from .scene import reading_key
    zones.sort(key=lambda z: reading_key(z["polygon"], sc.height))
    for n, z in enumerate(zones, 1):
        zid = f"zone{n}"
        sc.objects.append(Element(zid, "zone", z["polygon"], 0.8, source="legend"))
        sc.meta["zones"][zid] = {"label": z["label"], "color": z["color"]}
    # чертёжный лист без подписей помещений (трубопроводы, конструкции): помещения по стенам, рамки от VLM
    # на мелком масштабе (общежитие A-2, 9.9 px/м: комната ~30 px) рамки ИИ неточны, и он объединяет
    # повторяющиеся комнаты в одну рамку на крыло — помещения по стенам только от WALL_ROOMS_MIN_PPM
    drawing_sheet = bool(re.match(r"^[A-Z]{1,3}[-.]?\d", str(tb.get("sheet_number") or "")))
    # чертёжный лист: мелкие объекты (двери, окна, стены, сантехника) — YOLO по тайлам в масштабе обучения
    # (fp/tiles.py; школа 1:200: стены 3 -> 70 участков, двери-кандидаты 10/32 -> 32/32 на eval/doors_gt.json)
    full_img, ff = (full, f) if full is not None and f else (img, 1.0)
    if est.get("px_per_m") and drawing_sheet and os.environ.get("FP_TILES", "1") != "0" and sc.meta.get("model"):
        from .tiles import retile
        sc.meta["tiles"] = retile(sc, full_img, ff, sc.meta["model"])
    if (est.get("px_per_m") or 0) >= WALL_ROOMS_MIN_PPM and tb.get("sheet_number") and not sheet.get("area_labels"):
        rooms_by_walls(img, sc, sheet)
    # чертёжный лист (номер вида M-101A, A-2, C01): двери ещё и по дугам открывания — модель обучена на квартирах
    # и на таких листах видит их слабо. На квартирных планах дуги вредят (53.7% -> 36.6% на train+val eval_set),
    # поэтому там не включаются. Мелкий масштаб (дверь < 20 px, школа 1:200) — тоже нет: под четверть окружности
    # такого размера подходят стулья и буквы
    # двери на чертёжном листе: кандидаты с запасом (тайлы + все дуги) -> проверка ИИ по вырезкам
    # (fp/door_verify.py: на eval/doors_gt.json лишних 5 -> 1, пропущено 22 -> 0, ~$0.04 на лист).
    # Без ключа или при ошибке — прежний путь: дуги рядом со стенами
    verified = False
    if est.get("px_per_m") and drawing_sheet and os.environ.get("FP_DOOR_VERIFY", "1") != "0" and sc.meta.get("model"):
        from .door_verify import apply as verify_doors
        try:
            dv = verify_doors(sc, full_img, ff, sc.meta["model"])
            sc.meta["door_verify"] = dv
            sc.meta["sheet_cost_usd"] = round(sc.meta.get("sheet_cost_usd", 0) + dv["cost_usd"], 6)
            verified = True
        except Exception as e:  # noqa: BLE001
            sc.meta["door_verify"] = {"done": False, "error": f"{type(e).__name__}: {e}"}
    if not verified and (est.get("px_per_m") or 0) >= ARC_MIN_PPM and drawing_sheet:
        add_arc_doors(img, sc)
    # окна по полосам остекления в стенах — тоже только на чертёжных листах: на квартирах CubiCasa модель
    # по окнам лучше, а полосы добавляют лишние (train+val eval_set: 41.5% -> 24.5%)
    if est.get("px_per_m") and drawing_sheet:
        add_glazing_windows(img, sc)
    # лист трубопроводов: подписи диаметров и марки (ИИ), трубы по линиям (fp.piping)
    title = str(tb.get("drawing_title") or "").upper()
    if (est.get("px_per_m") or 0) >= ARC_MIN_PPM and drawing_sheet and \
            ("PIPING" in title or "PIPE" in title or str(tb.get("sheet_number") or "").upper().startswith(("M", "P"))):
        add_piping(img, sc)
    # помещения: площади и плинтусы (подписи + заливка по стенам + контуры модели)
    from .spaces import build as build_spaces
    build_spaces(img, sc)
    # опись плана: один вызов сильной модели — сколько чего по 12 типам вопросов (fp/inventory.py);
    # на какие вопросы она отвечает, а где только сверяет YOLO — fp/route.json
    if os.environ.get("FP_INVENTORY", "1") != "0":
        from .inventory import read_inventory
        try:
            inv, c3 = read_inventory(img)
            inv["cost_usd"] = round(c3, 6)
        except Exception as e:  # noqa: BLE001 — без описи ответы идут от YOLO
            inv, c3 = {"error": f"{type(e).__name__}: {e}"}, 0.0
        sc.meta["inventory"] = inv
        sc.meta["sheet_cost_usd"] = round(sc.meta.get("sheet_cost_usd", 0) + c3, 6)
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


_ROOM_WORDS = {"room_bath": ("toilet", "restroom", "bath", "wc", "lavatory", "shower", "туалет", "санузел"),
               "room_storage": ("storage", "store", "closet", "janitor", "кладов"),
               "room_entry": ("corridor", "hall", "vestibule", "lobby", "entry", "коридор", "холл"),
               "room_kitchen": ("kitchen", "кухня")}


def _room_cls(name: str) -> str:
    n = name.lower()
    return next((c for c, words in _ROOM_WORDS.items() if any(w in n for w in words)), "room_other")


def rooms_by_walls(img: np.ndarray, sc: Scene, sheet: dict) -> list[dict]:
    """Помещения листа, где модель их не видит (обучена на квартирах), а подписей комнат нет.
    Один узкий вызов VLM даёт рамки помещений, код прижимает их к линиям стен и уточняет заливкой
    (fp.rooms_cv.rooms_from_bbox). Площадь и периметр считает код, не VLM. Помещения модели на таком листе —
    в основном шум (графы штампа, площадки снаружи), поэтому они заменяются; их id — в meta."""
    from .rooms_cv import barrier_mask, rooms_from_bbox
    from .scene import Element, reading_key
    from .sheet import find_rooms
    try:
        found, cost = find_rooms(img, sheet.get("drawing_bbox"))
    except Exception as e:  # noqa: BLE001
        sc.meta["wall_rooms_note"] = f"VLM не ответила: {e}"
        return []
    sc.meta["sheet_cost_usd"] = round(sc.meta.get("sheet_cost_usd", 0) + cost, 6)
    if not found:
        sc.meta["wall_rooms"] = []
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bar = barrier_mask(gray)
    ppm = sc.meta["px_per_m"]
    rows = []
    dropped = []
    for rm in found:
        if re.search(r"\btypical\b|\(typ", rm["name"], re.I):   # «Patient room (typical)» — рамка на группу комнат
            dropped.append(rm["name"])
            continue
        parts = rooms_from_bbox(gray, rm["bbox"], ppm, barrier=bar)
        for i, p in enumerate(parts, 1):
            name = rm["name"] if len(parts) == 1 else f"{rm['name']} · {i}"
            rows.append({**p, "name": name, "named": rm["named"], "vlm_bbox": [round(v) for v in rm["bbox"]]})
    # одна и та же область из двух рамок (VLM повторилась или заливка вышла за рамку) — оставляем первую
    from shapely.geometry import Polygon
    keep = []
    for r in rows:
        a = Polygon(r["polygon"]).buffer(0)
        if all(a.intersection(Polygon(k["polygon"]).buffer(0)).area < 0.5 * min(a.area, Polygon(k["polygon"]).area)
               for k in keep):
            keep.append(r)
    if not keep:  # всё отброшено — помещения модели остаются как есть
        sc.meta["wall_rooms"] = []
        sc.meta["wall_rooms_dropped"] = dropped
        return []
    keep.sort(key=lambda r: reading_key(r["polygon"], sc.height))
    sc.meta["yolo_rooms_replaced"] = [r.id for r in sc.rooms]
    counters: dict = {}
    sc.rooms = []
    for r in keep:
        cls = _room_cls(r["name"])
        counters[cls] = counters.get(cls, 0) + 1
        r["id"] = f"{cls.removeprefix('room_')}_{counters[cls]}"
        # уверенность: точный контур заливкой или рамка, прижатая к стенам со всех сторон — средняя
        r["confidence"] = "medium" if r["how"] in ("fill", "split") or (r["how"] == "bbox" and r["sides_snapped"] == 4) \
            else "low"
        conf = {"medium": 0.75, "low": 0.5}[r["confidence"]]
        sc.rooms.append(Element(r["id"], cls, [[float(x), float(y)] for x, y in r["polygon"]], conf, source="vlm+walls"))
    sc.meta["wall_rooms"] = keep
    if dropped:
        sc.meta["wall_rooms_dropped"] = dropped
    sc.assign_rooms()
    return keep


def add_arc_doors(img: np.ndarray, sc: Scene) -> dict:
    """Второй способ найти двери — дуги открывания (fp.doors_cv). Дверь модели, подтверждённая дугой, —
    уверенность выше; дуга без двери модели на границе помещения — новая дверь (source="arc")."""
    from shapely.geometry import Polygon
    from .doors_cv import find_door_arcs, merge_with_model, near_walls
    from .scene import Element
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ppm = sc.meta.get("px_per_m")
    doors = [o for o in sc.objects if o.cls == "door" and o.source != "arc"]
    door_px = float(np.median([max(np.ptp(np.array(d.polygon), 0)) for d in doors])) if doors else None
    if not ppm and not door_px:
        info = {"done": False, "note": "нет масштаба и дверей модели — радиус дуги неизвестен"}
        sc.meta["door_arcs"] = info
        return info
    region = (sc.meta.get("sheet") or {}).get("drawing_bbox")
    arcs = find_door_arcs(gray, ppm, door_px, region)
    eff_ppm = ppm or door_px / 0.9
    walls = [r.shape() for r in sc.rooms] + [o.shape() for o in sc.objects if o.cls == "wall"]
    walls = [Polygon(w.exterior) if w.geom_type == "Polygon" else w for w in walls]
    kept = near_walls(arcs, walls, eff_ppm)
    new, confirmed = merge_with_model(kept, doors, eff_ppm)
    sc.objects = [o for o in sc.objects if not (o.cls == "door" and o.source == "arc")]
    for d in doors:
        if d.id in confirmed:
            d.conf = max(d.conf, 0.9)
    n0 = sum(1 for o in sc.objects if o.cls == "door")
    for i, a in enumerate(new, 1):
        x0, y0, x1, y1 = a["bbox"]
        sc.objects.append(Element(f"da{i}", "door", [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                  0.6, source="arc"))
    sc.assign_rooms()
    info = {"done": True, "arcs": len(arcs), "near_walls": len(kept), "confirmed": sorted(confirmed),
            "added": len(new), "model_doors": n0}
    sc.meta["door_arcs"] = info
    return info


def add_glazing_windows(img: np.ndarray, sc: Scene) -> dict:
    """Второй способ найти окна — полосы остекления в стенах (fp.windows_cv.find_glazing). Окно модели,
    совпавшее с полосой, — подтверждено; полоса без окна модели — новое окно (source="glazing")."""
    from .rooms_cv import wall_mask
    from .scene import Element
    from .windows_cv import find_glazing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ppm = sc.meta.get("px_per_m")
    if not ppm:
        ppm = wall_mask(gray)[1] / 0.25          # без масштаба: толстая стена ≈ 25 см
    bands = find_glazing(gray, ppm)
    region = (sc.meta.get("sheet") or {}).get("drawing_bbox")
    if region:
        bands = [b for b in bands if region[0] <= (b["bbox"][0] + b["bbox"][2]) / 2 <= region[2]
                 and region[1] <= (b["bbox"][1] + b["bbox"][3]) / 2 <= region[3]]
    sc.objects = [o for o in sc.objects if not (o.cls == "window" and o.source == "glazing")]
    wins = [o for o in sc.objects if o.cls == "window"]
    pad = 0.2 * ppm
    confirmed, n_new = set(), 0
    for b in bands:
        x0, y0, x1, y1 = b["bbox"]
        hit = next((o.id for o in wins
                    if min(p[0] for p in o.polygon) - pad <= x1 and x0 <= max(p[0] for p in o.polygon) + pad
                    and min(p[1] for p in o.polygon) - pad <= y1 and y0 <= max(p[1] for p in o.polygon) + pad), None)
        if hit:
            confirmed.add(hit)
            continue
        n_new += 1
        sc.objects.append(Element(f"wg{n_new}", "window", [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.6,
                                  source="glazing"))
    for o in wins:
        if o.id in confirmed:
            o.conf = max(o.conf, 0.9)
    sc.assign_rooms()
    info = {"done": True, "bands": len(bands), "confirmed": sorted(confirmed), "added": n_new,
            "model_windows": len(wins)}
    sc.meta["window_glazing"] = info
    return info


def add_piping(img: np.ndarray, sc: Scene) -> dict:
    """Трубы по диаметрам и арматура по маркам (fp.piping). Подписи читает ИИ по фрагментам рамки чертежа,
    расширенной вправо: подписи главных коллекторов стоят на полях."""
    from .piping import build
    from .scene import Element
    from .sheet import read_piping
    h, w = img.shape[:2]
    bb = (sc.meta.get("sheet") or {}).get("drawing_bbox") or [0, 0, w, h]
    region = [max(0, bb[0] - 0.03 * w), max(0, bb[1] - 0.05 * h), min(w, bb[2] + 0.12 * w), min(h, bb[3] + 0.05 * h)]
    try:
        items, cost = read_piping(img, region)
    except Exception as e:  # noqa: BLE001
        sc.meta["piping"] = {"error": f"{type(e).__name__}: {e}"}
        return sc.meta["piping"]
    p = build(img, sc, items)
    p["cost_usd"] = round(cost, 6)
    p["items"] = items
    sc.meta["piping"] = p
    sc.meta["sheet_cost_usd"] = round(sc.meta.get("sheet_cost_usd", 0) + cost, 6)
    sc.objects = [o for o in sc.objects if o.cls not in ("pipe", "fitting")]
    for s in p["segments"]:
        x0, y0, x1, y1 = s["bbox"]
        sc.objects.append(Element(s["id"], "pipe", [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.6, source="piping"))
    n = 0
    for f in p["fittings"]:
        for x, y in f["points"]:
            n += 1
            f.setdefault("ids", []).append(f"f{n}")
            sc.objects.append(Element(f"f{n}", "fitting", [[x - 6, y - 6], [x + 6, y - 6], [x + 6, y + 6], [x - 6, y + 6]],
                                      0.7, source="label"))
    sc.assign_rooms()
    return p
