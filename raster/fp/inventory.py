"""Опись плана: один вызов сильной модели на план при загрузке — сколько чего на плане по 12 типам.

Зачем: YOLO (обучена на CubiCasa) хорошо считает окна и двери, но путает раковины с унитазами, не знает
класса «душ», не различает WC и ванную, не читает подписи комнат (MH, KH, SR…) и этажи. Модель, которая
смотрит на картинку целиком, на этих типах заметно точнее (замер на eval_set Stoa). Цена растёт с числом
файлов, а не вопросов: опись делается один раз, вопросы потом отвечаются из сцены за $0.

Какой источник отвечает на какой тип — fp/query.py (ROUTE), решено по планам train+val, отчёт по test.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .sheet import _png

INV_MODEL = os.environ.get("FP_INVENTORY_MODEL", "claude-opus-5-5")
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
SIDE = 1568

# ключ описи -> что это (определения в промпте)
KEYS = {
    "windows": "windows in the outer walls; one window = one opening, even if it has mullions/several panes",
    "doors": "doors: doorways fitted with a door leaf (drawn as a leaf with a swing arc, or a sliding/folding door). "
             "Open doorways without a door leaf do not count",
    "toilets": "toilet bowls / WC seats",
    "sinks": "sinks and wash basins, kitchen and bathroom combined; a double-bowl sink counts as one",
    "showers_and_bathtubs": "showers plus bathtubs in total (a shower tray/cubicle or a bathtub drawn as a fixture, "
                            "or a room or area marked shower/SH/suihku)",
    "saunas": "saunas (rooms marked S, SAUNA, LÖYLYH, BASTU, or a room with sauna benches/stove)",
    "kitchens": "kitchens, including kitchenettes, open kitchens and utility kitchens (KEITTIÖ, K, KK, KT, KEITTOKOMERO, KÖK)",
    "bedrooms": "bedrooms (MH, makuuhuone, SOV, sovrum, bedroom). A walk-in closet (VH, vaatehuone) is NOT a bedroom",
    "wc_rooms": "separate WC rooms (small rooms marked WC that contain a toilet but are not a bathroom)",
    "bathrooms": "bathrooms / washrooms (KH, KPH, PH, PESUH, SH, badrum, bathroom), not counting separate WC rooms",
    "balconies": "balconies (parveke, P, PARV, lasitettu parveke = glazed balcony, balkong). Terraces on the ground "
                 "are not balconies unless marked as such",
    "floors": "floors (storeys) shown in the image: count separate floor-plan drawings of different storeys "
              "(e.g. 1. KERROS and 2. KERROS); a single plan is 1",
}

SCHEMA = {"type": "object", "properties": {
    **{k: {"type": "integer", "description": v} for k, v in KEYS.items()},
    "room_labels": {"type": "array", "items": {"type": "string"},
                    "description": "every room label text you read on the plan, as written"},
    "notes": {"type": "string", "description": "one short line: what was hard to count, if anything"}},
    "required": [*KEYS, "room_labels", "notes"], "additionalProperties": False}

PROMPT = """This image is an architectural floor plan (often Finnish or Swedish, sometimes rotated text or two storeys \
on one image). Count each item below exactly as defined. Look at every room; read the room labels first (they may be \
rotated 90 or 180 degrees). Count fixtures from their drawn symbols. Give integers; 0 when there is none.

{defs}

Common abbreviations: MH makuuhuone = bedroom; OH olohuone = living room; K/KK/KT keittiö = kitchen; \
TK/TUP = dining/living; ET eteinen = hall; KH kylpyhuone = bathroom; PH/PESUH pesuhuone = washroom; \
S/SAUNA/LÖYLYH = sauna; WC = toilet room; VH vaatehuone = walk-in closet; VAR = storage; \
KHH kodinhoitohuone = utility room (not a bathroom unless marked with shower); P/PARV = balcony; \
TERASSI = terrace; SR sovrum = bedroom; KÖK = kitchen; BADRUM/BAD = bathroom; TV/TVÄTT = laundry."""


def read_inventory(img: np.ndarray) -> tuple[dict, float]:
    """Один вызов модели на план. Возвращает ({ключ: число, room_labels, notes, model}, стоимость $)."""
    import anthropic
    data, _ = _png(img, SIDE)
    defs = "\n".join(f"- {k}: {v}" for k, v in KEYS.items())
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
               {"type": "text", "text": PROMPT.format(defs=defs)}]
    client = anthropic.Anthropic()
    kw = dict(model=INV_MODEL, max_tokens=16000, messages=[{"role": "user", "content": content}],
              output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
    if INV_MODEL.startswith("claude-opus") or INV_MODEL.startswith("claude-sonnet"):
        # Opus 5.5: мышление не отключается — глубина только через effort; отказ классификатора -> fallback
        resp = client.beta.messages.create(**kw, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    else:
        resp = client.messages.create(**kw)
    pin, pout = PRICES.get(INV_MODEL, (4.0, 20.0))
    cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
    if resp.stop_reason == "refusal":
        return {"model": INV_MODEL, "error": "refusal"}, cost
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return {"model": INV_MODEL, "error": f"нет ответа ({resp.stop_reason})"}, cost
    inv = json.loads(text)
    inv["model"] = INV_MODEL
    return inv, cost
