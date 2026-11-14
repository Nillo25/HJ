"""Масштаб плана: сколько пикселей в метре (px_per_m). Без него площади только в px².

Источники (по убыванию надёжности):
  manual      — пользователь отметил на плане две точки и ввёл расстояние;
  scale_bar   — линейная шкала (0'—80'): VLM читает подпись и примерное место, длину в пикселях меряем сами;
  area_labels — подписанные площади комнат («MH 10,6 m²»): VLM читает подпись и примерное место,
                комнату находим по нашему графу, px_per_m = sqrt(площадь_px / площадь_m2), медиана по комнатам.
VLM вызывается один раз на план (при загрузке), не на каждый вопрос.
"""
from __future__ import annotations

import base64
import json
import math
import os
import statistics

import cv2
import numpy as np

from .scene import Scene

UNIT_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "in": 0.0254}
VLM_MODEL = os.environ.get("FP_VLM_MODEL", "claude-haiku-5-5")
VLM_MAX_SIDE = 1568  # длинная сторона картинки для VLM; координаты пересчитываем обратно
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


# ---------------- ручной ----------------
def from_two_points(p1, p2, length: float, unit: str = "m") -> dict:
    px = math.dist(p1, p2)
    m = length * UNIT_M[unit]
    if px <= 0 or m <= 0:
        raise ValueError("нужны две разные точки и положительная длина")
    return {"px_per_m": px / m, "source": "manual", "confidence": "high",
            "detail": f"{px:.0f} px = {length:g} {unit}"}


# ---------------- по подписям площадей ----------------
def from_area_labels(scene: Scene, labels: list[dict]) -> dict | None:
    """labels: [{"value_m2": 10.6, "x": .., "y": ..}] в пикселях исходной картинки."""
    from shapely.geometry import Point
    est, used = [], []
    for lb in labels:
        v = float(lb.get("value_m2") or 0)
        if not 0.5 <= v <= 500:
            continue
        pt = Point(lb["x"], lb["y"])
        rooms = [r for r in scene.rooms if r.shape().buffer(15).contains(pt)]
        if not rooms:
            continue
        room = min(rooms, key=lambda r: r.shape().area)  # самая маленькая содержащая — сама комната
        est.append(math.sqrt(room.shape().area / v))
        used.append({"room": room.id, "label_m2": v, "text": lb.get("text")})
    if not est:
        return None
    # median_low — одна из оценок: при чётном числе подписей среднее двух далёких значений не совпало бы
    # ни с одной, и «согласованных» не осталось бы (падение на high_quality_architectural-18)
    med = statistics.median_low(est)
    # согласованность: насколько оценки по разным комнатам совпадают
    good = [e for e in est if abs(e / med - 1) <= 0.15]
    spread = (max(good) / min(good) - 1) if len(good) > 1 else None
    conf = "high" if len(good) >= 3 and spread is not None and spread <= 0.15 else "medium" if len(good) >= 2 else "low"
    px = statistics.median(good)
    for u in used:
        u["poly_m2"] = round(next(r for r in scene.rooms if r.id == u["room"]).shape().area / px ** 2, 2)
    return {"px_per_m": px, "source": "area_labels", "confidence": conf,
            "detail": f"{len(good)} из {len(est)} подписей согласованы", "labels": used}


# ---------------- по линейной шкале ----------------
def measure_bar(img: np.ndarray, bbox) -> float | None:
    """Длина шкалы в пикселях внутри примерной рамки от VLM: самая длинная горизонтальная тёмная полоса."""
    h, w = img.shape[:2]
    pad = 25
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    dark = gray < 170  # линии шкалы бывают серыми (тонкий контур сегментов)
    # линии, проходящие через всю рамку от края до края (граница штампа, рамка листа), — не шкала:
    # шкала лежит внутри рамки VLM с запасом pad, а такая линия длиннее рамки и даёт её ширину (S-100F: 158 px вместо 98)
    edge = 3
    crossing = dark[:, :edge].any(1) & dark[:, -edge:].any(1) & (dark.sum(1) >= 0.9 * dark.shape[1])
    dark[crossing] = False
    cov = dark.sum(1)
    if cov.max() < 30:
        return None
    # полоса шкалы: строки вокруг самой «чёрной», где тёмного не меньше половины от максимума
    r = int(cov.argmax())
    top = bot = r
    while top > 0 and cov[top - 1] >= 0.5 * cov[r]:
        top -= 1
    while bot < len(cov) - 1 and cov[bot + 1] >= 0.5 * cov[r]:
        bot += 1
    # сегменты шкалы бывают «ступеньками» на разной высоте — расширяем полосу по вертикали до 6 строк,
    # но только по сплошным непустым строкам: подписи 0/500/1000/2000 отделены пустой строкой и в длину не входят
    for _ in range(6):
        if top > 0 and cov[top - 1] > 0:
            top -= 1
        if bot < len(cov) - 1 and cov[bot + 1] > 0:
            bot += 1
    xs = np.flatnonzero(dark[top:bot + 1].any(0))
    # чередование чёрных/белых сегментов даёт разрывы — допускаем их, но не дальше 15% ширины рамки
    max_gap = max(10, int(0.15 * (x1 - x0)))
    best = cur = (xs[0], xs[0])
    for x in xs[1:]:
        cur = (cur[0], x) if x - cur[1] <= max_gap else (x, x)
        if cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    length = best[1] - best[0] + 1
    return float(length) if length >= 30 else None


def from_scale_bar(img: np.ndarray, bar: dict) -> dict | None:
    px = measure_bar(img, bar["bbox"])
    unit = bar.get("unit", "m")
    if not px or unit not in UNIT_M or not bar.get("length"):
        return None
    m = float(bar["length"]) * UNIT_M[unit]
    return {"px_per_m": px / m, "source": "scale_bar", "confidence": "medium",
            "detail": f"шкала {bar['length']:g} {unit} = {px:.0f} px"}


# ---------------- по тексту масштаба и формату листа ----------------
PAPER_MM = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (297, 210),
            "ARCH E": (1219, 914), "ARCH D": (914, 610), "ARCH C": (610, 457), "ANSI E": (1118, 864),
            "ANSI D": (864, 559), "ANSI C": (559, 432), "ANSI B": (432, 279)}


def ratio_from_text(text: str) -> float | None:
    """'1:200' -> 200; '1/4" = 1'-0"' -> 48; '1" = 40'-0"' -> 480."""
    import re
    t = (text or "").replace(" ", "")
    m = re.search(r"1:(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)?/?(\d+)?\"=(\d+)'(?:-(\d+)\")?", t)  # дюймы на листе = футы-дюймы в натуре
    if m:
        a, b = m.group(1), m.group(2)
        paper_in = (float(a) / float(b)) if a and b else float(a or 1)
        real_in = float(m.group(3)) * 12 + float(m.group(4) or 0)
        return real_in / paper_in if paper_in else None
    return None


def from_text_and_paper(img: np.ndarray, scale_text: str, paper: str) -> dict | None:
    """Растровая картинка ЦЕЛОГО листа: ширина в пикселях = ширина формата. Тогда 1:N даёт px/м."""
    n = ratio_from_text(scale_text)
    key = (paper or "").upper().replace("-", " ").strip()
    if not n or key not in PAPER_MM:
        return None
    h, w = img.shape[:2]
    pw, ph = PAPER_MM[key]
    paper_w_mm = pw if w >= h else ph
    px_per_paper_m = w / (paper_w_mm / 1000)
    return {"px_per_m": px_per_paper_m / n, "source": "scale_text", "confidence": "low",
            "detail": f"{scale_text} на {key}: только если картинка — весь лист без полей"}


# ---------------- VLM: один вызов на план ----------------
SCHEMA = {
    "type": "object",
    "properties": {
        "scale_bar": {"anyOf": [{"type": "null"}, {
            "type": "object",
            "properties": {"length": {"type": "number"}, "unit": {"type": "string", "enum": list(UNIT_M)},
                           "bbox": {"type": "array", "items": {"type": "number"}}},
            "required": ["length", "unit", "bbox"], "additionalProperties": False}]},
        "area_labels": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "value_m2": {"type": "number"},
                           "x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["text", "value_m2", "x", "y"], "additionalProperties": False}},
    },
    "required": ["scale_bar", "area_labels"], "additionalProperties": False,
}

PROMPT = (
    "This is an architectural floor plan image of {w}x{h} pixels (x to the right, y down).\n"
    "1. Room area labels: list every label written inside a room that states its floor area "
    "(e.g. '10,6 m²', '12.5 m2', 'MH 10,6', '150 sq ft' converted to m²). value_m2 = the area in square metres. "
    "x, y = pixel coordinates of the CENTER of that label text.\n"
    "2. Scale bar: if there is a graphic scale bar (a ruler with marks like 0' 20' 40' 80' or 0 1 2 5 m), "
    "give its total length as labelled at its far end, the unit (m, cm, mm, ft, in) and bbox = [x0, y0, x1, y1] "
    "of the bar itself in pixels. Otherwise null. A text-only scale like 1:100 is NOT a scale bar.\n"
    "Only report what is clearly readable; do not guess."
)


def vlm_read(img: np.ndarray) -> tuple[dict, float]:
    import anthropic
    h, w = img.shape[:2]
    k = min(1.0, VLM_MAX_SIDE / max(h, w))
    small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    ok, buf = cv2.imencode(".png", small)
    data = base64.standard_b64encode(buf.tobytes()).decode()
    sh, sw = small.shape[:2]
    resp = anthropic.Anthropic().messages.create(
        model=VLM_MODEL, max_tokens=4000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
            {"type": "text", "text": PROMPT.format(w=sw, h=sh)}]}],
    )
    pin, pout = PRICES.get(VLM_MODEL, (4.0, 20.0))
    cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
    if resp.stop_reason == "refusal":
        return {"scale_bar": None, "area_labels": []}, cost
    out = json.loads(next(b.text for b in resp.content if b.type == "text"))
    s = 1 / k  # обратно в пиксели исходной картинки
    for lb in out["area_labels"]:
        lb["x"], lb["y"] = lb["x"] * s, lb["y"] * s
    if out.get("scale_bar") and len(out["scale_bar"].get("bbox", [])) == 4:
        out["scale_bar"]["bbox"] = [v * s for v in out["scale_bar"]["bbox"]]
    elif out.get("scale_bar"):
        out["scale_bar"] = None
    return out, cost


def estimate(img: np.ndarray, scene: Scene, read: dict | None = None, cost: float = 0.0) -> dict:
    """Масштаб для сцены. read — уже прочитанный лист (fp.sheet.read_sheet); без него VLM вызывается здесь.
    Возвращает meta-поле scale (или причину, почему масштаба нет)."""
    if read is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return {"px_per_m": None, "source": None, "detail": "нет ключа для VLM — задайте масштаб вручную"}
        try:
            read, cost = vlm_read(img)
        except Exception as e:  # noqa: BLE001
            return {"px_per_m": None, "source": None, "detail": f"VLM не ответила: {e}"}
    cands = []
    if read.get("scale_bar"):
        r = from_scale_bar(img, read["scale_bar"])
        if r:
            cands.append(r)
    r = from_area_labels(scene, read.get("area_labels", []))
    if r:
        cands.append(r)
    r = from_text_and_paper(img, read.get("scale_text", ""), read.get("paper_size", ""))
    if r:
        cands.append(r)
    if not cands:
        return {"px_per_m": None, "source": None, "detail": "на плане не нашлось шкалы или подписанных площадей",
                "vlm_cost_usd": round(cost, 6)}
    best = choose(cands, from_dimensions(scene, read))
    best["vlm_cost_usd"] = round(cost, 6)
    return best


# ---------------- выбор между источниками ----------------
AGREE, DISAGREE = 0.05, 0.10  # совпали: ≤5% — high; разошлись: >10% — нужен третий источник


def _rel(a: dict, b: dict) -> float:
    return abs(a["px_per_m"] / b["px_per_m"] - 1)


def from_dimensions(scene: Scene, read: dict) -> dict | None:
    """Грубая проверка по габаритному размеру: самый большой размер с листа (27000 мм) против длинной стороны
    охвата найденных объектов внутри рамки чертежа. Самостоятельным масштабом не выбирается — только судья,
    когда шкала, подписи площадей и текст масштаба расходятся (S-100F: 1318 px / 27 м = 48,8 px/м)."""
    dims = [float(d.get("value_m") or 0) for d in (read.get("dimensions") or [])]
    dims = [d for d in dims if d > 0]
    if not dims or not scene.objects:
        return None
    db = read.get("drawing_bbox")
    boxes = []
    for o in scene.objects:
        s = o.shape()
        if s.is_empty:
            continue
        c = s.centroid
        if db and len(db) == 4 and not (db[0] <= c.x <= db[2] and db[1] <= c.y <= db[3]):
            continue
        boxes.append(s.bounds)
    if not boxes:
        return None
    b = np.array(boxes)
    span = max(b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min())
    m = max(dims)
    return {"px_per_m": span / m, "source": "dimensions", "confidence": "low",
            "detail": f"охват объектов {span:.0f} px / габарит {m:g} м"}


def choose(cands: list[dict], dims: dict | None = None) -> dict:
    """Лучший кандидат по уверенности; два источника в пределах 5% — high. Если лучший расходится с другим
    источником больше чем на 10%, не выбираем его молча: берём тот, что сходится (≤10%) с третьим
    (другой источник или габаритный размер dims); если третьего нет или он не решает — уверенность low."""
    rank = {"high": 2, "medium": 1, "low": 0}
    best = max(cands, key=lambda c: rank[c["confidence"]])
    fmt = lambda c: f"{c['source']} {c['px_per_m']:.1f}"
    agree = [c for c in cands if c is not best and _rel(c, best) <= AGREE]
    if agree:
        return {**best, "confidence": "high", "detail": best["detail"] + f"; совпало с: {agree[0]['source']}"}
    conflicts = [c for c in cands if c is not best and _rel(c, best) > DISAGREE]
    if not conflicts:
        return dict(best)
    judges = cands + ([dims] if dims else [])
    support = {id(c): [j for j in judges if j is not c and j["source"] != c["source"] and _rel(j, c) <= DISAGREE]
               for c in cands}
    backed = [c for c in cands if support[id(c)]]
    gap = "источники разошлись: " + ", ".join(fmt(c) for c in cands) + " px/м"
    if backed:
        win = max(backed, key=lambda c: (len(support[id(c)]), rank[c["confidence"]]))
        rivals = [c for c in backed if c is not win and _rel(c, win) > DISAGREE]
        if not rivals:
            by = ", ".join(fmt(j) for j in support[id(win)])
            conf = "medium" if rank[win["confidence"]] < 2 else win["confidence"]
            return {**win, "confidence": conf, "detail": win["detail"] + f"; {gap}; выбран по согласию с: {by}"}
    return {**best, "confidence": "low", "detail": best["detail"] + f"; {gap}; проверьте масштаб вручную"}
