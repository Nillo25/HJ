"""Второе мнение VLM по сантехнике («арбитр» из плана Stoa). Один вызов на план: вырезаем санузлы, кухни и места,
где модель нашла сантехнику, VLM перечисляет приборы.

Замер на тестовых планах CubiCasa (scripts в истории сессии):
  * ПРАВИТЬ детекции по VLM нельзя — подсчёт ухудшается (унитазы 88% -> 70–78%, Haiku и Sonnet): модель,
    обученная на этих планах, различает значки лучше универсальной VLM;
  * как СИГНАЛ полезно: VLM согласна с числом модели — модель верна в 83%, не согласна — в 64%.
Поэтому check_fixtures только считает приборы по мнению VLM; query.count понижает уверенность при расхождении.
"""
from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np

from .scene import Element, Scene

FIXTURES = ["toilet", "sink", "shower", "bathtub"]
VLM_MODEL = os.environ.get("FP_VLM_MODEL", "claude-haiku-5-5")
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
CROP_SIDE = 640
MAX_CROPS = 8

SCHEMA = {"type": "object", "properties": {"crops": {"type": "array", "items": {"type": "object", "properties": {
    "crop": {"type": "integer"},
    "fixtures": {"type": "array", "items": {"type": "object", "properties": {
        "cls": {"type": "string", "enum": FIXTURES},
        "bbox": {"type": "array", "items": {"type": "number"}, "description": "[x0,y0,x1,y1] in that crop's pixels"}},
        "required": ["cls", "bbox"], "additionalProperties": False}}},
    "required": ["crop", "fixtures"], "additionalProperties": False}}},
    "required": ["crops"], "additionalProperties": False}

PROMPT = (
    "Each image is a crop of an architectural floor plan (top view). For EVERY crop list the plumbing fixture "
    "symbols that are clearly drawn in it:\n"
    "- toilet: a WC bowl — an oval/egg-shaped bowl attached to a small rectangular cistern;\n"
    "- sink: a washbasin or kitchen sink — a round/oval/rectangular basin, often with a small tap dot, no cistern;\n"
    "- shower: a shower tray — a square with an X or a drain dot, or a shower area;\n"
    "- bathtub: a long rectangular tub with rounded inner outline.\n"
    "bbox = the symbol only, in that crop's own pixel coordinates. Do not list furniture, stoves, appliances, "
    "sauna benches or text. Return an entry for every crop, with an empty list if it has none.")


def _box(poly) -> list[float]:
    p = np.asarray(poly, np.float32)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def regions(scene: Scene, w: int, h: int) -> list[list[int]]:
    """Где искать: комнаты санузлов/кухонь + рамки вокруг найденной моделью сантехники; слитые, с запасом."""
    boxes = [_box(r.polygon) for r in scene.rooms if r.cls in ("room_bath", "room_kitchen")]
    boxes += [_box(o.polygon) for o in scene.objects if o.cls in FIXTURES]
    pad = 0.04 * max(w, h)
    boxes = [[max(0, b[0] - pad), max(0, b[1] - pad), min(w, b[2] + pad), min(h, b[3] + pad)] for b in boxes]
    merged = [list(b) for b in boxes]
    changed = True
    while changed:  # сливаем пересекающиеся рамки
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    merged[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    merged.sort(key=lambda m: -(m[2] - m[0]) * (m[3] - m[1]))
    return [[int(v) for v in m] for m in merged[:MAX_CROPS]]


def vlm_fixtures(img: np.ndarray, crops: list[list[int]]) -> tuple[list[tuple[str, list[float]]], float]:
    """Один вызов: все вырезки -> [(класс, бокс в пикселях всего плана)]."""
    import anthropic
    content, scales = [], []
    for i, (x0, y0, x1, y1) in enumerate(crops, 1):
        c = img[y0:y1, x0:x1]
        k = CROP_SIDE / max(c.shape[:2])
        c = cv2.resize(c, (round(c.shape[1] * k), round(c.shape[0] * k)), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        scales.append((k, x0, y0))
        content.append({"type": "text", "text": f"Crop {i}: {c.shape[1]}x{c.shape[0]} px"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.standard_b64encode(cv2.imencode('.png', c)[1].tobytes()).decode()}})
    content.append({"type": "text", "text": PROMPT})
    resp = anthropic.Anthropic().messages.create(
        model=VLM_MODEL, max_tokens=3000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}])
    pin, pout = PRICES.get(VLM_MODEL, (4.0, 20.0))
    cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
    if resp.stop_reason == "refusal":
        return [], cost
    out = []
    for c in json.loads(next(b.text for b in resp.content if b.type == "text"))["crops"]:
        if not 1 <= c["crop"] <= len(scales):
            continue
        k, x0, y0 = scales[c["crop"] - 1]
        for f in c["fixtures"]:
            if len(f["bbox"]) == 4:
                bx = f["bbox"]
                out.append((f["cls"], [x0 + bx[0] / k, y0 + bx[1] / k, x0 + bx[2] / k, y0 + bx[3] / k]))
    # одна и та же раковина на стыке двух вырезок — оставляем один раз
    uniq: list[tuple[str, list[float]]] = []
    for cls, b in out:
        if not any(c2 == cls and _iou(b, b2) > 0.3 for c2, b2 in uniq):
            uniq.append((cls, b))
    return uniq, cost


def _near(a, b) -> float:
    """Похожесть по месту для неточных рамок VLM: 1 — центры совпали, 0 — центр дальше размера прибора."""
    ca, cb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    return max(0.0, 1 - np.hypot(ca[0] - cb[0], ca[1] - cb[1]) / (0.8 * size))


def arbitrate(img: np.ndarray, scene: Scene) -> dict:
    """Сверяет сантехнику модели с VLM и правит сцену. Возвращает отчёт для meta['fixture_check']."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return {"done": False, "reason": "нет ключа ИИ"}
    h, w = img.shape[:2]
    crops = regions(scene, w, h)
    if not crops:
        return {"done": True, "crops": 0, "note": "на плане нет санузлов, кухонь и сантехники"}
    try:
        vlm, cost = vlm_fixtures(img, crops)
    except Exception as e:  # noqa: BLE001
        return {"done": False, "reason": f"VLM не ответила: {e}"}
    inside = lambda b: any(c[0] <= (b[0] + b[2]) / 2 <= c[2] and c[1] <= (b[1] + b[3]) / 2 <= c[3] for c in crops)
    model = [o for o in scene.objects if o.cls in FIXTURES]
    used, report = set(), {"confirmed": 0, "relabeled": [], "added": [], "unconfirmed": []}
    for o in model:
        ob = _box(o.polygon)
        best = max(((i, _near(ob, vb)) for i, (_, vb) in enumerate(vlm) if i not in used), key=lambda t: t[1], default=(None, 0))
        if best[0] is not None and best[1] > 0:
            used.add(best[0])
            vcls = vlm[best[0]][0]
            if vcls == o.cls:
                o.conf = max(o.conf, 0.95)  # два независимых источника согласны
                o.source = f"{o.source}+vlm"
                report["confirmed"] += 1
            else:
                report["relabeled"].append(f"{o.id}: {o.cls} → {vcls}")
                o.cls, o.conf, o.source = vcls, 0.75, f"{o.source}→vlm"
        elif inside(ob):
            o.conf = min(o.conf, 0.5)  # VLM смотрела это место и прибора не увидела
            report["unconfirmed"].append(o.id)
    for i, (cls, b) in enumerate(vlm):
        if i in used:
            continue
        n = sum(1 for o in scene.objects if o.cls == cls) + 1
        prefix = {"toilet": "t", "sink": "s", "shower": "sh", "bathtub": "bt"}[cls]
        eid = f"{prefix}v{n}"
        poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        scene.objects.append(Element(eid, cls, poly, 0.7, source="vlm"))
        report["added"].append(f"{eid}: {cls}")
    # перенумеровать id по новым классам не нужно: id стабильны, класс — в поле cls
    scene.assign_rooms()
    report.update({"done": True, "crops": len(crops), "vlm_fixtures": len(vlm), "cost_usd": round(cost, 6)})
    return report


def check_fixtures(img: np.ndarray, scene: Scene) -> dict:
    """Второе мнение: число приборов каждого класса по VLM (сцену не меняет). -> meta['fixture_check']."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return {"done": False, "reason": "нет ключа ИИ"}
    h, w = img.shape[:2]
    crops = regions(scene, w, h)
    try:
        vlm, cost = vlm_fixtures(img, crops) if crops else ([], 0.0)
    except Exception as e:  # noqa: BLE001
        return {"done": False, "reason": f"VLM не ответила: {e}"}
    return {"done": True, "model": VLM_MODEL, "crops": crops, "counts": {c: sum(v[0] == c for v in vlm) for c in FIXTURES},
            "boxes": [{"cls": c, "bbox": [round(x, 1) for x in b]} for c, b in vlm], "cost_usd": round(cost, 6)}
