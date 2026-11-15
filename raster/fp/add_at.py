"""Кнопка «+ Add»: пользователь кликнул по пропущенному объекту — что там и какой у него контур.

1. Наша модель (YOLO) заново смотрит вырезку вокруг клика с низким порогом: объект часто был найден, но отброшен
   как пограничный. Детекция, которая содержит клик (или ближайшая к нему), даёт класс и точный контур. $0.
2. Модель ничего не видит -> Haiku по той же вырезке: какой объект в центре (красный крестик) и его рамка.
Добавление — fp/corrections.add_object (журнал правок, отменяется undo).
"""
from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np

ADD_CLASSES = ["window", "door", "toilet", "sink", "bathtub", "shower", "sauna_bench"]
HAIKU = os.environ.get("FP_ADD_MODEL", "claude-haiku-5-5")
PRICE = (0.10, 0.50)  # $/1M вход/выход, Haiku 5.5
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["cls", "box"],
          "properties": {"cls": {"type": "string", "enum": ADD_CLASSES + ["none"]},
                         "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}}
PROMPT = ("This is a crop of an architectural floor plan. The red cross marks where the user clicked: they say an object "
          "there was missed. Which object is under or right next to the cross: " + ", ".join(ADD_CLASSES) +
          "? Give its bounding box [x0, y0, x1, y1] in pixels of this crop, tight around the symbol. "
          "A window is a thin glazing band in an outer wall; a door is a leaf with its opening arc. "
          "If nothing like that is there, cls = none.")


def _crop(img: np.ndarray, x: float, y: float, half: int):
    h, w = img.shape[:2]
    x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
    x1, y1 = min(w, int(x) + half), min(h, int(y) + half)
    return img[y0:y1, x0:x1], x0, y0


def _by_model(img, x, y, half, model_path, want=None):
    from .infer import THRESHOLDS, detect_yolo
    crop, ox, oy = _crop(img, x, y, half)
    th = json.loads(json.dumps(THRESHOLDS))
    for k, v in th.items():
        if not k.startswith("_"):
            v["conf"] = 0.05
    best = None
    for cls, poly, conf in detect_yolo(crop, model_path, imgsz=640, thresholds=th):
        if cls not in ADD_CLASSES or (want and cls != want):
            continue
        p = np.asarray(poly, dtype=float) + (ox, oy)
        inside = cv2.pointPolygonTest(p.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), True)
        if inside < -half * 0.35:  # далеко от клика — не то, что имел в виду пользователь
            continue
        key = (inside >= 0, conf)
        if best is None or key > best[0]:
            best = (key, cls, p.round(1).tolist(), conf)
    return (best[1], best[2], round(best[3], 2)) if best else None


def _by_haiku(img, x, y, half, want=None):
    import anthropic
    crop, ox, oy = _crop(img, x, y, half)
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop.copy()
    cx, cy, s = int(x) - ox, int(y) - oy, max(4, half // 12)
    cv2.line(vis, (cx - s, cy), (cx + s, cy), (0, 0, 255), 2)
    cv2.line(vis, (cx, cy - s), (cx, cy + s), (0, 0, 255), 2)
    ok, png = cv2.imencode(".png", vis)
    r = anthropic.Anthropic().messages.create(
        model=HAIKU, max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(png.tobytes()).decode()}},
            {"type": "text", "text": PROMPT if not want else (
                f"This is a crop of an architectural floor plan. The red cross marks where the user clicked; the user "
                f"says there is a {want} there that was missed. Give the bounding box [x0, y0, x1, y1] of that {want} "
                f"in pixels of this crop, tight around the symbol. cls = {want}; cls = none only if no such symbol is "
                f"anywhere near the cross.")}]}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
    cost = (r.usage.input_tokens * PRICE[0] + r.usage.output_tokens * PRICE[1]) / 1e6
    if r.stop_reason == "refusal":
        return None, cost
    a = json.loads(next(b.text for b in r.content if b.type == "text"))
    if a.get("cls") not in ADD_CLASSES:
        return None, cost
    bx0, by0, bx1, by1 = a["box"]
    bx0, bx1 = sorted((max(0, bx0), min(crop.shape[1], bx1)))
    by0, by1 = sorted((max(0, by0), min(crop.shape[0], by1)))
    if bx1 - bx0 < 3 or by1 - by0 < 3:  # рамки нет — маленький квадрат вокруг клика
        bx0, by0, bx1, by1 = cx - s, cy - s, cx + s, cy + s
    poly = [[bx0 + ox, by0 + oy], [bx1 + ox, by0 + oy], [bx1 + ox, by1 + oy], [bx0 + ox, by1 + oy]]
    return (a["cls"], [[round(v, 1) for v in p] for p in poly]), cost


def detect_at(img: np.ndarray, x: float, y: float, ppm: float | None, model_path: str | None,
              want: str | None = None) -> dict:
    """{cls, polygon, how ('model' | 'haiku' | 'user'), conf, cost_usd}. want = тип со слов пользователя: ищем только
    его контур; не нашли ни модель, ни Haiku -> polygon None (рамка среднего размера). Без want: ValueError."""
    h, w = img.shape[:2]
    half = int(max(48, min(400, (ppm or 0) * 1.5 or max(w, h) * 0.04)))
    if model_path:
        hit = _by_model(img, x, y, half, model_path, want)
        if hit:
            return {"cls": hit[0], "polygon": hit[1], "how": "model", "conf": hit[2], "cost_usd": 0.0}
    try:
        found, cost = _by_haiku(img, x, y, half, want)
    except Exception:  # noqa: BLE001 — нет ключа/сети: тип уже известен, рамка среднего размера
        if not want:
            raise
        found, cost = None, 0.0
    if want and (not found or found[0] != want):
        return {"cls": want, "polygon": None, "how": "user", "conf": 1.0, "cost_usd": round(cost, 6)}
    if not found:
        raise ValueError(f"не понял, что здесь: ни модель, ни ИИ не видят окна, двери или сантехники (${cost:.4f})")
    return {"cls": found[0], "polygon": found[1], "how": "haiku", "conf": 1.0, "cost_usd": round(cost, 6)}
