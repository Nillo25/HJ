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


def _tiles(H: int, W: int, m: int, core: int = CORE):
    for y0 in range(0, H, core):
        for x0 in range(0, W, core):
            cx1, cy1 = min(W, x0 + core), min(H, y0 + core)
            yield (x0, y0, cx1, cy1), (max(0, x0 - m), max(0, y0 - m), min(W, cx1 + m), min(H, cy1 + m))


def _tiled_points(full: np.ndarray, ppm_full: float, region, *, work_ppm: float, core: int, margin_m: float,
                  model: str, prompt: str, schema: dict) -> tuple[list, list, int, int, float]:
    """Лист -> фрагменты в масштабе work_ppm -> вызов model по каждому -> точки items / room_labels в px full,
    только из ядра своего фрагмента. Возвращает (points, labels, фрагментов, без ответа, цена $)."""
    import anthropic
    s = min(1.0, work_ppm / ppm_full)   # мельче листа не увеличиваем; координаты пересчитываются тем же s
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
    m = int(round(margin_m * work_ppm))
    jobs = []
    for c, (tx0, ty0, tx1, ty1) in _tiles(H, W, m, core):
        t = g[ty0:ty1, tx0:tx1]
        if (t < 200).mean() < 0.003:   # пустая бумага
            continue
        jobs.append((c, (tx0, ty0), t))
    client = anthropic.Anthropic()
    pin, pout = PRICES.get(model, (4.0, 20.0))
    strong = model.startswith(("claude-opus", "claude-sonnet"))

    def run(job):
        c, (tx0, ty0), t = job
        data = base64.standard_b64encode(cv2.imencode(".png", t)[1].tobytes()).decode()
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                   {"type": "text", "text": prompt.format(w=t.shape[1], h=t.shape[0])}]
        kw = dict(model=model, max_tokens=8000, messages=[{"role": "user", "content": content}],
                  output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
        if strong:  # как в fp/inventory.py: отказ классификатора -> fallback
            r = client.beta.messages.create(**kw, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            r = client.messages.create(**kw)
        usd = (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
        if r.stop_reason == "refusal":
            return c, (tx0, ty0), None, usd
        text = next((b.text for b in r.content if b.type == "text"), None)
        return c, (tx0, ty0), (json.loads(text) if text else None), usd

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
    return points, labels, len(jobs), failed, cost


def read_inventory_tiled(full: np.ndarray, ppm_full: float, region=None) -> tuple[dict, float]:
    """full — лист в полном разрешении (после поворота/обрезки), ppm_full — его px/м, region — рамка чертежа
    в px full. Возвращает (опись в формате fp/inventory + points/room_labels в px full, цена $)."""
    prompt = PROMPT.replace("{w}", "{{w}}").replace("{h}", "{{h}}").format(**KEYS)
    points, labels, n, failed, cost = _tiled_points(full, ppm_full, region, work_ppm=WORK_PPM, core=CORE,
                                                    margin_m=MARGIN_M, model=TILE_MODEL, prompt=prompt, schema=SCHEMA)
    inv = {k: 0 for k in KEYS if k != "floors"}
    for p in points:
        inv[TYPES[p["type"]]] += 1
    inv["floors"] = 1   # один лист — один план этажа; несколько этажей на листе тайлы не различают
    inv.update(model=TILE_MODEL, method="tiles", tiles=n, tiles_failed=failed, work_ppm=WORK_PPM,
               points=points, room_labels=[lb["text"] for lb in labels], room_label_points=labels,
               notes=f"по {n} фрагментам листа в {WORK_PPM:g} px/м" + (f", без ответа: {failed}" if failed else ""))
    return inv, cost


# --- второй проход: сильная модель, только комнаты, крупные фрагменты ---
# Мелочь (двери, окна, сантехника) на большом листе отвечают YOLO + проверка Haiku; Opus здесь читает только подписи
# комнат — они крупнее значков, поэтому масштаб вдвое мельче (пикселей вчетверо меньше) и фрагменты крупнее.
# НЕ ЗАМЕРЕНО: эталона комнат по большим листам нет (тест вслепую 15.11.2026). Выключить: FP_INV_ROOMS=0.
ROOM_MODEL = os.environ.get("FP_INV_ROOM_MODEL", "claude-opus-5-5")
ROOM_WORK_PPM = float(os.environ.get("FP_INV_ROOM_PPM", "40"))
ROOM_CORE = 1200          # px в масштабе ROOM_WORK_PPM; фрагмент с запасом 1200 + 2*120 = 1440 <= 1568 (fp/inventory.SIDE)
ROOM_MARGIN_M = 3.0       # подпись комнаты длиннее значка: запас больше, чтобы подпись на стыке была видна целиком
ROOM_TYPES = {"sauna": "saunas", "kitchen": "kitchens", "bedroom": "bedrooms", "wc_room": "wc_rooms",
              "bathroom": "bathrooms", "balcony": "balconies"}
ROOM_SCHEMA = {"type": "object", "properties": {
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "type": {"type": "string", "enum": sorted(ROOM_TYPES)},
        "x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["type", "x", "y"], "additionalProperties": False}},
    "room_labels": SCHEMA["properties"]["room_labels"]},
    "required": ["items", "room_labels"], "additionalProperties": False}
ROOM_PROMPT = """This image is one fragment ({{w}}x{{h}} px, x right, y down) of a large architectural drawing sheet. List EVERY ROOM of the types below whose name label you can see in it, one entry per room, with the pixel position of the centre of its name label. Rooms are recognised by their labels (they may be rotated); a label cut by the fragment edge: list it too. Do not list doors, windows or fixtures. Do not guess; an empty list is a valid answer.

Types:
- sauna: {saunas}
- kitchen: {kitchens}
- bedroom: {bedrooms}
- wc_room: {wc_rooms}
- bathroom: {bathrooms}
- balcony: {balconies}

Common abbreviations: MH makuuhuone = bedroom; K/KK/KT keittiö = kitchen; KH kylpyhuone = bathroom; PH/PESUH pesuhuone = washroom; S/SAUNA/LÖYLYH = sauna; WC = toilet room; VH vaatehuone = walk-in closet (not a bedroom); KHH = utility room (not a bathroom unless marked with shower); P/PARV = balcony; SR sovrum = bedroom; KÖK = kitchen; BADRUM/BAD = bathroom.

Also list every room name label you can read (room_labels: text as printed, centre of the text)."""


def read_rooms_tiled(full: np.ndarray, ppm_full: float, region=None) -> tuple[dict, float]:
    """Комнаты большого листа сильной моделью по крупным фрагментам. -> ({ключ комнаты: число, ...}, цена $)."""
    prompt = ROOM_PROMPT.format(**KEYS)
    points, labels, n, failed, cost = _tiled_points(full, ppm_full, region, work_ppm=ROOM_WORK_PPM, core=ROOM_CORE,
                                                    margin_m=ROOM_MARGIN_M, model=ROOM_MODEL, prompt=prompt,
                                                    schema=ROOM_SCHEMA)
    rooms = {k: 0 for k in ROOM_TYPES.values()}
    for p in points:
        rooms[ROOM_TYPES[p["type"]]] += 1
    return {"counts": rooms, "model": ROOM_MODEL, "tiles": n, "tiles_failed": failed, "work_ppm": ROOM_WORK_PPM,
            "points": points, "room_labels": [lb["text"] for lb in labels], "cost_usd": round(cost, 6)}, cost
