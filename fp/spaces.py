"""Помещения листа: площадь и погонные метры плинтуса — из нескольких независимых источников.

Для каждой подписанной комнаты (подписи читает VLM при загрузке, fp.sheet):
  label — площадь, напечатанная на плане («KEITTIÖ 11.2 m²»): это ответ, если она есть;
  fill  — заливка свободного пола от подписи до стен (fp.rooms_cv, без нейросети, любой лист);
  yolo  — контур комнаты от YOLO-seg, внутри которого стоит подпись (планы вроде CubiCasa).
Уверенность: два источника совпали в пределах 10% — high; один — medium; разошлись — low.

Плинтус = периметр упрощённого контура (без зазубрин растра и ниш мебели) минус ширина дверей, касающихся
комнаты. Если площадь подписана, контур масштабируется под неё: форма — с плана, размер — из подписи.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from .rooms_cv import spaces_from_labels
from .scene import Element, Scene

AGREE = 0.10       # подпись на плане vs геометрия
AGREE_GEO = 0.15   # заливка vs контур модели: заливка идёт до грани стены, маска модели чуть ужата


def _simplified_perimeter(poly: list, px_per_m: float) -> float:
    c = np.array(poly, np.float32).reshape(-1, 1, 2)
    # почти прямоугольная комната (заполняет описанный прямоугольник на 85%+) — периметр прямоугольника:
    # ступеньки растра, ниши мебели и выступы заливки не раздувают длину плинтуса
    (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
    if rw * rh > 0 and cv2.contourArea(c) / (rw * rh) >= 0.85:
        return 2 * (rw + rh) / px_per_m
    eps = 0.12 * px_per_m  # 12 см: убирает ступеньки растра и мелкие ниши
    s = cv2.approxPolyDP(c, eps, True)
    return cv2.arcLength(s, True) / px_per_m


def _doors_touching(poly: list, doors: list[list[float]], px_per_m: float) -> list[float]:
    shp = Polygon(poly).buffer(0.15 * px_per_m)
    out = []
    for x0, y0, x1, y1 in doors:
        if shp.intersects(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])):
            out.append(max(x1 - x0, y1 - y0) / px_per_m)  # бокс двери = створка × створка
    return out


def build(img: np.ndarray, scene: Scene) -> list[dict]:
    """Заполняет scene.meta['spaces'] и элементы class='space' (для подсветки). Нужны масштаб и подписи."""
    px = scene.meta.get("px_per_m")
    sheet = scene.meta.get("sheet") or {}
    labels = [lb for lb in sheet.get("room_labels") or sheet.get("area_labels") or [] if lb.get("text")]
    scene.objects = [o for o in scene.objects if o.cls != "space"]
    if not px or not labels:
        scene.meta["spaces"] = []
        scene.meta["spaces_note"] = "нужны масштаб и подписи помещений на плане" if labels else "на плане нет подписей помещений"
        return []
    doors = []
    for o in scene.objects:
        if o.cls == "door":
            p = np.array(o.polygon)
            doors.append([*p.min(0), *p.max(0)])
    fills = spaces_from_labels(img, labels, px, doors)
    # каждая комната модели — одной подписи: ближайшей к её центру
    owner: dict = {}
    for r in scene.rooms:
        shp = r.shape()
        inside = [(Point(lb["x"], lb["y"]).distance(shp.centroid), i) for i, lb in enumerate(labels, 1)
                  if shp.buffer(10).contains(Point(lb["x"], lb["y"]))]
        if inside:
            owner[r.id] = min(inside)[1]
    spaces = []
    for n, (lb, f) in enumerate(zip(labels, fills), 1):
        label_m2 = float(lb.get("value_m2") or 0) or None
        pt = Point(lb["x"], lb["y"])
        yolo = [r for r in scene.rooms if r.shape().buffer(10).contains(pt)]
        yolo = min(yolo, key=lambda r: r.shape().area) if yolo else None
        src = {}
        # заливка «застряла» в мелкой ячейке (штриховка, текст) и подтвердить её нечем — не считаем
        if f.get("ok") and (f["area_m2"] >= 1.0 or label_m2):
            src["fill"] = (f["area_m2"], f["polygon"])
        # контур модели — только если эта область достаётся этой подписи (ближайшей к центру области)
        if yolo is not None and owner.get(yolo.id) == n:
            src["yolo"] = (round(yolo.shape().area / px ** 2, 2), yolo.polygon)
        if not src and not label_m2:
            spaces.append({"id": None, "name": lb["text"], "ok": False,
                           "note": "контур не замкнут (проём без двери) и площадь не подписана"})
            continue
        # геометрия: ближайшая к подписи, иначе заливка (не зависит от обученной модели)
        if label_m2 and src:
            geo = min(src, key=lambda k: abs(src[k][0] / label_m2 - 1))
        else:
            geo = "fill" if "fill" in src else ("yolo" if src else None)
        geo_m2, poly = src[geo] if geo else (None, None)
        area = label_m2 or geo_m2
        # согласие независимых источников
        vals = [("label", label_m2)] + [(k, v[0]) for k, v in src.items()]
        vals = [(k, v) for k, v in vals if v]

        def ok_pair(a, b):
            tol = AGREE if "label" in (a[0], b[0]) else AGREE_GEO
            return abs(a[1] / b[1] - 1) <= tol
        agree = any(ok_pair(a, b) for i, a in enumerate(vals) for b in vals[i + 1:])
        spread = len(vals) >= 2 and not agree
        conf = "high" if agree else ("low" if spread else "medium")
        base = door_w = base_note = None
        n_doors = 0
        if poly:
            per = _simplified_perimeter(poly, px)
            if label_m2 and geo_m2:
                per *= math.sqrt(label_m2 / geo_m2)  # форма с плана, размер из подписи
            dw = _doors_touching(poly, doors, px)
            n_doors, door_w = len(dw), round(sum(dw), 2)
            base = round(max(0.0, per - sum(dw)), 2)
            # правдоподобие: периметр комнаты редко больше 1.5 периметра квадрата той же площади
            if area and per / (4 * math.sqrt(area)) > 1.5:
                conf = "low"
                base_note = "контур неровный — плинтус, вероятно, завышен"
        sid = f"sp{n}"
        if poly:
            scene.objects.append(Element(sid, "space", poly, 0.9 if conf == "high" else 0.75 if conf == "medium" else 0.5,
                                         source=geo))
        spaces.append({"id": sid if poly else None, "name": lb["text"], "ok": True, "area_m2": round(area, 2),
                       "area_source": "подпись на плане" if label_m2 else {"fill": "заливка по стенам", "yolo": "контур модели"}[geo],
                       "label_m2": label_m2, "fill_m2": src.get("fill", (None,))[0], "yolo_m2": src.get("yolo", (None,))[0],
                       "baseboard_m": base, "baseboard_note": base_note, "doors": n_doors, "door_width_m": door_w,
                       "confidence": conf})
    scene.meta["spaces"] = spaces
    return spaces
