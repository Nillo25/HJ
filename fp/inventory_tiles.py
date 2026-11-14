"""Опись большого чертёжного листа по тайлам: лист в полном разрешении режется на фрагменты с запасом,
ИИ по каждому фрагменту перечисляет объекты С КООРДИНАТАМИ, объект засчитывается тайлу, в чьём ядре лежит его
точка — на перекрытиях ничего не считается дважды.

Зачем: обычная опись (fp/inventory.py) смотрит на весь лист, уменьшенный до 1568 px. На школе 14000 px дверь —
несколько пикселей: опись видела 75 дверей из ~130, 3 раковины из ~14, и расхождение с детекциями делало
низкой уверенность всех ответов листа. Координаты дают и замер по эталону (eval/doors_gt.json), и подсветку.
"""
from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .inventory import KEYS

TILE_MODEL = os.environ.get("FP_INV_TILE_MODEL", "claude-haiku-5-5")
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
WORK_PPM = 80.0       # масштаб фрагмента: при 45 px/м Haiku не видел двери M-101A (0 из 6); 80 — школа 21/26 без лишних
CORE = 800            # ядро тайла, px в масштабе WORK_PPM (школа: 37 фрагментов, $0.018)
MARGIN_M = 1.5        # запас вокруг ядра: объект на стыке целиком виден хотя бы в одном тайле
WORKERS = 4
MIN_SIDE = 3000       # тайлы — только для листов крупнее (иначе обычная опись по целой картинке)

# тип точки -> ключ описи (fp/inventory.KEYS)
TYPES = {"door": "doors", "window": "windows", "toilet": "toilets", "sink": "sinks", "shower": "showers_and_bathtubs",
         "bathtub": "showers_and_bathtubs", "sauna": "saunas", "kitchen": "kitchens", "bedroom": "bedrooms",
         "wc_room": "wc_rooms", "bathroom": "bathrooms", "balcony": "balconies"}

SCHEMA = {"type": "object", "properties": {
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "type": {"type": "string", "enum": sorted(TYPES)},
        "x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["type", "x", "y"], "additionalProperties": False}},
    "room_labels": {"type": "array", "items": {"type": "object", "properties": {
        "text": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["text", "x", "y"], "additionalProperties": False}}},
    "required": ["items", "room_labels"], "additionalProperties": False}

PROMPT = """This image is one fragment ({w}x{h} px, x right, y down) of a large architectural drawing sheet. \
List EVERY item of the types below that you can see in it, one entry per item, with the pixel position of its centre \
(for a room type: the centre of its name label). Items cut by the fragment edge: list them too. Do not guess items \
you cannot see; an empty list is a valid answer.

Types:
- door: {doors}
- window: {windows}
- toilet: {toilets}
- sink: {sinks}
- shower: a shower tray/cubicle symbol or an area marked shower
- bathtub: a bathtub symbol
- sauna: {saunas}
- kitchen: {kitchens}
- bedroom: {bedrooms}
- wc_room: {wc_rooms}
- bathroom: {bathrooms}
- balcony: {balconies}

Also list every room name label you can read (room_labels: text as printed, centre of the text)."""


def _tiles(H: int, W: int, m: int):
    for y0 in range(0, H, CORE):
        for x0 in range(0, W, CORE):
            cx1, cy1 = min(W, x0 + CORE), min(H, y0 + CORE)
            yield (x0, y0, cx1, cy1), (max(0, x0 - m), max(0, y0 - m), min(W, cx1 + m), min(H, cy1 + m))


def read_inventory_tiled(full: np.ndarray, ppm_full: float, region=None) -> tuple[dict, float]:
    """full — лист в полном разрешении (после поворота/обрезки), ppm_full — его px/м, region — рамка чертежа
    в px full. Возвращает (опись в формате fp/inventory + points/room_labels в px full, цена $)."""
    import anthropic
    s = WORK_PPM / ppm_full
    if region:
        x0, y0, x1, y1 = (int(v) for v in region)
        x0, y0 = max(0, x0), max(0, y0)
        full = full[y0:y1, x0:x1]
    else:
        x0 = y0 = 0
    g = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else full
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    m = int(round(MARGIN_M * WORK_PPM))
    jobs = []
    for core, (tx0, ty0, tx1, ty1) in _tiles(H, W, m):
        t = g[ty0:ty1, tx0:tx1]
        if (t < 200).mean() < 0.003:   # пустая бумага
            continue
        jobs.append((core, (tx0, ty0), t))
    defs = {k: v for k, v in KEYS.items()}
    client = anthropic.Anthropic()
    pin, pout = PRICES.get(TILE_MODEL, (4.0, 20.0))

    def run(job):
        core, (tx0, ty0), t = job
        data = base64.standard_b64encode(cv2.imencode(".png", t)[1].tobytes()).decode()
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                   {"type": "text", "text": PROMPT.format(w=t.shape[1], h=t.shape[0], **defs)}]
        r = client.messages.create(model=TILE_MODEL, max_tokens=8000, messages=[{"role": "user", "content": content}],
                                   output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
        usd = (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
        if r.stop_reason == "refusal":
            return core, (tx0, ty0), None, usd
        text = next((b.text for b in r.content if b.type == "text"), None)
        return core, (tx0, ty0), (json.loads(text) if text else None), usd

    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(run, jobs))
    cost = sum(r[3] for r in res)
    points, labels, failed = [], [], 0
    for (cx0, cy0, cx1, cy1), (tx0, ty0), ans, _ in res:
        if ans is None:
            failed += 1
            continue
        for kind, lst in (("items", points), ("room_labels", labels)):
            for it in ans.get(kind) or []:
                gx, gy = tx0 + it["x"], ty0 + it["y"]
                if cx0 <= gx < cx1 and cy0 <= gy < cy1:     # точка в ядре этого тайла
                    lst.append({**it, "x": round(x0 + gx / s, 1), "y": round(y0 + gy / s, 1)})
    inv = {k: 0 for k in KEYS if k != "floors"}
    for p in points:
        inv[TYPES[p["type"]]] += 1
    inv["floors"] = 1   # один лист — один план этажа; несколько этажей на листе тайлы не различают
    inv.update(model=TILE_MODEL, method="tiles", tiles=len(jobs), tiles_failed=failed, work_ppm=WORK_PPM,
               points=points, room_labels=[lb["text"] for lb in labels], room_label_points=labels,
               notes=f"по {len(jobs)} фрагментам листа в {WORK_PPM:g} px/м" + (f", без ответа: {failed}" if failed else ""))
    return inv, cost
