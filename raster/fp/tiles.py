"""YOLO по тайлам на чертёжных листах: модель обучена на квартирах CubiCasa, где дверь/окно — десятки пикселей.
Целый лист ужимается до 2048 px (школа 1:200 на A2 — 17.6 px/м), и мелкие объекты пропадают. Здесь лист
приводится к TILE_PPM (~55 px/м — масштаб, на котором модель видит сантехнику: замер stoa, TEST-4),
режется на ядра CORE px с запасом MARGIN_M со всех сторон; объект засчитывается тайлу, в чьё ядро попал его
центр — без дублей и без разрезанных объектов (запас >= половины самого крупного объекта: двустворчатая дверь).

Замер (eval/doors_gt.json, 3 листа, 32 двери, scripts/exp_doors.py): двери-кандидаты по тайлам находят 32/32
против 10/32 у прохода по целому листу; на школе стены 3 -> 70 участков, окна 15 -> 19.
Комнаты сюда не входят: они крупнее запаса и резались бы — остаются от прохода по целому листу.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

TILE_PPM = float(os.environ.get("FP_TILE_PPM", "55"))
CORE = 1024
MARGIN_M = 1.5
TILE_CLASSES = {"wall", "door", "window", "toilet", "sink", "bathtub", "shower", "sauna_bench"}


def full_res(original: np.ndarray, info: dict) -> np.ndarray:
    """Исходная картинка с теми же поворотом и обрезкой, что сделал preprocess (без уменьшения)."""
    img = original
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[:, :, 3:] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    ang = info.get("deskew_deg") or 0.0
    if abs(ang) > 0.2:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    if info.get("crop"):
        x0, y0, x1, y1 = info["crop"]
        img = img[y0:y1, x0:x1]
    return img


def detect_tiles(img: np.ndarray, ppm: float, model: str, thresholds: dict | None = None,
                 candidates: list | None = None) -> list[tuple[str, np.ndarray, float]]:
    """img — лист в полном разрешении, ppm — его px/м. Возвращает [(cls, polygon в px img, conf)] для TILE_CLASSES."""
    from .infer import detect_yolo
    s = TILE_PPM / ppm
    g = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    H, W = g.shape[:2]
    m = int(round(MARGIN_M * TILE_PPM))
    out = []
    for y0 in range(0, H, CORE):
        for x0 in range(0, W, CORE):
            cx1, cy1 = min(W, x0 + CORE), min(H, y0 + CORE)
            tx0, ty0, tx1, ty1 = max(0, x0 - m), max(0, y0 - m), min(W, cx1 + m), min(H, cy1 + m)
            tile = g[ty0:ty1, tx0:tx1]
            if (cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) < 200).mean() < 0.002:  # пустая бумага
                continue
            cand: list = []
            for cls, poly, conf in detect_yolo(tile, model, thresholds=thresholds, candidates=cand):
                if cls not in TILE_CLASSES:
                    continue
                p = np.asarray(poly, dtype=float) + (tx0, ty0)
                c = p.mean(0)
                if x0 <= c[0] < cx1 and y0 <= c[1] < cy1:
                    out.append((cls, p / s, conf))
            if candidates is not None:
                for cls, conf, cx, cy in cand:
                    gx, gy = cx + tx0, cy + ty0
                    if x0 <= gx < cx1 and y0 <= gy < cy1:
                        candidates.append([cls, conf, gx / s, gy / s])
    return out


def retile(sc, full: np.ndarray, f: float, model: str) -> dict:
    """Заменить в сцене мелкие объекты прохода по целому листу на объекты по тайлам.
    full — лист в полном разрешении; f — коэффициент full -> координаты сцены."""
    from .scene import PREFIX, Element
    ppm_scene = sc.meta.get("px_per_m")
    if not ppm_scene:
        return {"done": False, "why": "нет масштаба"}
    ppm_full = ppm_scene / f
    import json as _json
    from .door_verify import CAND_CONF
    from .infer import THRESHOLDS
    th = _json.loads(_json.dumps(THRESHOLDS))
    door_thr = th["door"]["conf"]
    th["door"]["conf"] = CAND_CONF  # один проход: уверенные двери -> сцена, остальные -> кандидаты для проверки ИИ
    # только рамка чертежа (+ запас): штамп, легенда, ключевая схема сбоку листа — не план; их объекты
    # убирает drop_outside_drawing до этого шага, и тайлы не должны возвращать их обратно (заодно быстрее)
    ox = oy = 0
    bb = (sc.meta.get("sheet") or {}).get("drawing_bbox")
    if bb and len(bb) == 4:
        mg = int(MARGIN_M * ppm_full)
        ox, oy = max(0, int(bb[0] / f) - mg), max(0, int(bb[1] / f) - mg)
        full = full[oy:int(bb[3] / f) + mg, ox:int(bb[2] / f) + mg]
    cands: list = []
    dets_all = [(c, p + (ox, oy), conf) for c, p, conf in
                detect_tiles(full, ppm_full, model, thresholds=th, candidates=cands)]
    cands = [[c, conf, x + ox, y + oy] for c, conf, x, y in cands]
    sc.door_candidates = [{"c": p.mean(0).tolist(), "box": [*p.min(0), *p.max(0)], "src": "yolo", "conf": float(c)}
                          for cls, p, c in dets_all if cls == "door"]
    dets = [d for d in dets_all if d[0] != "door" or d[2] >= door_thr]
    old = [o for o in sc.objects if o.source == "yolo" and o.cls in TILE_CLASSES]
    sc.objects = [o for o in sc.objects if not (o.source == "yolo" and o.cls in TILE_CLASSES)]
    counters: dict = {}
    for cls, p, conf in dets:
        counters[cls] = counters.get(cls, 0) + 1
        pts = [[round(float(x) * f, 1), round(float(y) * f, 1)] for x, y in p]
        sc.objects.append(Element(f"{PREFIX.get(cls, cls)}t{counters[cls]}", cls, pts, float(conf), source="yolo"))
    sc.assign_rooms()
    sc.meta["candidates"] = [[c, conf, round(x * f, 1), round(y * f, 1)] for c, conf, x, y in cands]
    return {"done": True, "ppm_full": round(ppm_full, 1), "tile_ppm": TILE_PPM, "region_full": [ox, oy, ox + full.shape[1], oy + full.shape[0]],
            "before": {c: sum(o.cls == c for o in old) for c in sorted(TILE_CLASSES)},
            "after": {c: counters.get(c, 0) for c in sorted(TILE_CLASSES)}}
