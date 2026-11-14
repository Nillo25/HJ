"""Чтение всего листа чертежа одним вызовом VLM при загрузке (не на каждый вопрос).

Что читаем: штамп (название, номер листа, дата, стадия), масштаб (текст + линейная шкала), указатель севера,
легенду (подпись + цвет образца), таблицы (подсчёты комнат, площади), общие примечания, габаритные размеры,
подписи площадей в комнатах. Цвета легенды потом ищем на самом плане -> зоны (find_zones), их можно подсветить
и посчитать площадь по масштабу.

Модель видит лист целиком (координаты) и 4 четверти крупнее (мелкий текст таблиц).
"""
from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np

UNITS = ["m", "cm", "mm", "ft", "in"]
VLM_MODEL = os.environ.get("FP_VLM_MODEL", "claude-haiku-5-5")
FULL_SIDE, TILE_SIDE = 1568, 1100
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}

_num = {"type": "number"}
_str = {"type": "string"}
_bbox = {"type": "array", "items": _num, "description": "[x0, y0, x1, y1] in FULL image pixels"}
SCHEMA = {
    "type": "object",
    "properties": {
        "title_block": {"type": "object", "properties": {
            "project": _str, "drawing_title": _str, "sheet_number": _str, "date": _str, "phase": _str,
            "organization": _str}, "required": ["project", "drawing_title", "sheet_number", "date", "phase",
                                                "organization"], "additionalProperties": False},
        "scale_text": {"type": "string", "description": "e.g. '1\" = 40\\'-0\"' or '1:100'; empty if none"},
        "scale_bar": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {
            "length": _num, "unit": {"type": "string", "enum": UNITS}, "bbox": _bbox},
            "required": ["length", "unit", "bbox"], "additionalProperties": False}]},
        "north": {"type": "string", "description": "where plan north points: up/down/left/right/other/none"},
        "drawing_bbox": _bbox,
        "paper_size": {"type": "string", "description": "sheet format if printed or evident: A0..A4, ARCH D, ANSI D...; empty if unknown"},
        "legend": {"type": "array", "items": {"type": "object", "properties": {
            "label": _str, "color_hex": _str, "swatch_bbox": _bbox},
            "required": ["label", "color_hex", "swatch_bbox"], "additionalProperties": False}},
        "tables": {"type": "array", "items": {"type": "object", "properties": {
            "title": _str, "rows": {"type": "array", "items": {"type": "object", "properties": {
                "label": _str, "value": _num, "unit": _str, "text": _str},
                "required": ["label", "value", "unit", "text"], "additionalProperties": False}}},
            "required": ["title", "rows"], "additionalProperties": False}},
        "notes": {"type": "array", "items": _str},
        "dimensions": {"type": "array", "items": {"type": "object", "properties": {
            "text": _str, "value_m": _num, "what": _str},
            "required": ["text", "value_m", "what"], "additionalProperties": False}},
        "area_labels": {"type": "array", "items": {"type": "object", "properties": {
            "text": _str, "value_m2": _num, "x": _num, "y": _num},
            "required": ["text", "value_m2", "x", "y"], "additionalProperties": False}},
    },
    "required": ["title_block", "scale_text", "scale_bar", "north", "drawing_bbox", "paper_size", "legend", "tables", "notes", "dimensions",
                 "area_labels"],
    "additionalProperties": False,
}

PROMPT = """You are reading one architectural drawing sheet. Image 1 is the WHOLE sheet, {w}x{h} pixels (x right, y down).
Images 2-5 are its four quarters (top-left, top-right, bottom-left, bottom-right) enlarged, only to help read small text.
ALL coordinates and bboxes you return must be in Image 1 pixels.

Extract, transcribing exactly what is printed (do not guess; leave a field empty / list empty if absent):
- title_block: project name, drawing title, sheet number, date, phase, organization.
- scale_text (e.g. 1" = 40'-0", 1:100) and scale_bar: a graphic ruler (0' 20' 40' 80' or 0 1 2 5 m) -> total length
  at its far end, unit, bbox of the bar itself. A text-only scale is not a scale bar.
- north: direction of the north arrow.
- drawing_bbox: bbox of the MAIN plan drawing only (the building with its dimension lines), excluding the title
  block, legend, tables, notes and any small key plan / inset.
- legend: every colour/hatch legend entry -> its label, the swatch colour as #rrggbb, swatch_bbox.
- tables: every schedule / table / list of values (room count, square footage, areas...) -> title and rows
  (label, numeric value, unit such as SF, m2, DORMS, pcs; text = the row exactly as printed).
- notes: general notes, each paragraph as one string.
- dimensions: overall / main dimension strings (e.g. 323'-4") with value_m in metres and what they measure.
- paper_size: the sheet format if printed in the title block (A1, A3, ARCH D, ...), else empty.
- area_labels: EVERY room name label written inside a room (e.g. 'KEITTIÖ 11.2 m2', 'CLASSROOM 3', 'WC').
  text = the label as printed; value_m2 = the room area in m2 if it is printed with the name, otherwise 0;
  x,y = the centre of the room NAME text."""


def _png(img: np.ndarray, side: int) -> tuple[str, float]:
    h, w = img.shape[:2]
    k = min(1.0, side / max(h, w))
    small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    return base64.standard_b64encode(cv2.imencode(".png", small)[1].tobytes()).decode(), k


def read_sheet(img: np.ndarray) -> tuple[dict, float]:
    """Один вызов VLM: лист целиком + 4 четверти. Возвращает (sheet, стоимость $)."""
    import anthropic
    h, w = img.shape[:2]
    full, k = _png(img, FULL_SIDE)
    imgs = [full]
    for y0, y1 in ((0, h // 2 + 40), (h // 2 - 40, h)):
        for x0, x1 in ((0, w // 2 + 40), (w // 2 - 40, w)):
            imgs.append(_png(img[y0:y1, x0:x1], TILE_SIDE)[0])
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": d}} for d in imgs]
    content.append({"type": "text", "text": PROMPT.format(w=round(w * k), h=round(h * k))})
    resp = anthropic.Anthropic().messages.create(
        model=VLM_MODEL, max_tokens=8000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}])
    pin, pout = PRICES.get(VLM_MODEL, (4.0, 20.0))
    cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
    if resp.stop_reason == "refusal":
        return {}, cost
    sheet = json.loads(next(b.text for b in resp.content if b.type == "text"))
    s = 1 / k  # координаты обратно в пиксели исходной картинки
    for lb in sheet["area_labels"]:
        lb["x"], lb["y"] = lb["x"] * s, lb["y"] * s
    for lg in sheet["legend"]:
        lg["swatch_bbox"] = [v * s for v in lg["swatch_bbox"]] if len(lg["swatch_bbox"]) == 4 else None
    db = sheet.get("drawing_bbox") or []
    sheet["drawing_bbox"] = [v * s for v in db] if len(db) == 4 else None
    if sheet.get("scale_bar"):
        bb = sheet["scale_bar"].get("bbox", [])
        sheet["scale_bar"] = {**sheet["scale_bar"], "bbox": [v * s for v in bb]} if len(bb) == 4 else None
    return sheet, cost


# ---------------- зоны по цветам легенды ----------------
def _hex(c: str):
    c = c.strip().lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) if len(c) == 6 else None


def swatch_color(img: np.ndarray, bbox, hint_hex: str) -> tuple[int, int, int] | None:
    """Точный цвет образца: самый частый «не белый и не чёрный» цвет внутри рамки (VLM даёт рамку примерно)."""
    if bbox:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        pad = 6
        crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].reshape(-1, 3)
        lum = crop.mean(1)
        crop = crop[(lum > 40) & (lum < 245)]
        if len(crop) >= 20:
            q = (crop // 8).astype(np.int32)
            keys, counts = np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], return_counts=True)
            sel = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2] == keys[counts.argmax()]
            b, g, r = np.median(crop[sel], 0)
            return int(r), int(g), int(b)
    return _hex(hint_hex)


def find_zones(img: np.ndarray, legend: list[dict], region=None, exclude: list | None = None,
               min_frac: float = 0.0003) -> list[dict]:
    """Для каждой записи легенды — области того же цвета внутри основного чертежа (region).
    Тонкие штрихи (текст, линии мебели, сглаженные края) сначала убираются, потом дыры закрываются.
    Близкие к серому цвета (CIRCULATION) ищутся строже: серый дают и края чёрного текста."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    allow = np.zeros((h, w), bool)
    if region and len(region) == 4:
        x0, y0, x1, y1 = [int(v) for v in region]
        allow[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    else:
        allow[:] = True
    for bb in exclude or []:
        x0, y0, x1, y1 = [int(v) for v in bb]
        allow[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = False
    k = max(2, int(round(min(h, w) / 600)))
    zones = []
    for lg in legend:
        rgb = swatch_color(img, lg.get("swatch_bbox"), lg.get("color_hex", ""))
        if not rgb:
            continue
        ref = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        chroma = float(np.hypot(ref[1] - 128, ref[2] - 128))
        tol = 5.0 if chroma < 6 else 10.0
        d = np.linalg.norm(lab - ref, axis=2)
        mask = ((d < tol) & allow).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2 * k + 1, 2 * k + 1), np.uint8))   # без текста и линий
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((6 * k + 1, 6 * k + 1), np.uint8))  # мебель внутри зоны
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_frac * h * w:
                continue
            poly = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(poly) >= 3:
                zones.append({"label": lg["label"], "color": "#%02x%02x%02x" % rgb, "polygon": poly.tolist(),
                              "area_px2": float(a)})
    return zones


BAR_SCHEMA = {"type": "object", "properties": {
    "found": {"type": "boolean"}, "tile": {"type": "integer", "description": "1-4"},
    "bbox": {"type": "array", "items": _num, "description": "[x0,y0,x1,y1] of the bar in THAT tile's pixels"},
    "length": _num, "unit": {"type": "string", "enum": UNITS}},
    "required": ["found", "tile", "bbox", "length", "unit"], "additionalProperties": False}


def find_scale_bar(img: np.ndarray) -> tuple[dict | None, float]:
    """Второй, узкий вызов VLM: только линейная шкала, по увеличенным четвертям листа (на целом листе её
    легко пропустить). Возвращает {"length","unit","bbox"} в пикселях исходной картинки или None."""
    import anthropic
    h, w = img.shape[:2]
    tiles = []
    for y0, y1 in ((0, h // 2 + 40), (h // 2 - 40, h)):
        for x0, x1 in ((0, w // 2 + 40), (w // 2 - 40, w)):
            data, k = _png(img[y0:y1, x0:x1], TILE_SIDE)
            tiles.append((data, k, x0, y0, round((x1 - x0) * k), round((y1 - y0) * k)))
    content = []
    for i, (data, k, x0, y0, tw, th) in enumerate(tiles, 1):
        content.append({"type": "text", "text": f"Tile {i}: {tw}x{th} px"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    content.append({"type": "text", "text": (
        "These are the four quarters of one drawing sheet (1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right). "
        "Is there a GRAPHIC scale bar (a ruler: a bar divided into segments with numbers such as 0 5 10 20 m or "
        "0' 10' 20' 40')? If yes: the tile it is in, bbox of the bar itself in that tile's pixels, the number at its "
        "far end (length) and the unit. A text-only scale such as 1:200 is not a scale bar.")})
    resp = anthropic.Anthropic().messages.create(
        model=VLM_MODEL, max_tokens=1000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": BAR_SCHEMA}},
        messages=[{"role": "user", "content": content}])
    pin, pout = PRICES.get(VLM_MODEL, (4.0, 20.0))
    cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
    if resp.stop_reason == "refusal":
        return None, cost
    r = json.loads(next(b.text for b in resp.content if b.type == "text"))
    if not r["found"] or not 1 <= r["tile"] <= 4 or len(r["bbox"]) != 4 or r["length"] <= 0:
        return None, cost
    _, k, x0, y0, _, _ = tiles[r["tile"] - 1]
    bx0, by0, bx1, by1 = r["bbox"]
    return {"length": r["length"], "unit": r["unit"],
            "bbox": [x0 + bx0 / k, y0 + by0 / k, x0 + bx1 / k, y0 + by1 / k]}, cost


# ---------------- помещения без подписей ----------------
ROOMS_SCHEMA = {"type": "object", "properties": {"rooms": {"type": "array", "items": {
    "type": "object", "properties": {
        "name": {"type": "string", "description": "name written in the room, else a short description"},
        "named_on_sheet": {"type": "boolean"},
        "bbox": {"type": "array", "items": _num, "description": "[x0,y0,x1,y1] of the room's inner wall faces"}},
    "required": ["name", "named_on_sheet", "bbox"], "additionalProperties": False}}},
    "required": ["rooms"], "additionalProperties": False}


def find_rooms(img: np.ndarray, region=None) -> tuple[list[dict], float]:
    """Узкий вызов VLM для листов без подписей помещений (трубопроводы, конструкции): рамки помещений здания.
    Модель YOLO обучена на квартирах и на таких листах помещений почти не видит, а стены и трубы нарисованы
    одинаковыми тонкими линиями — отличить их по геометрии нельзя. Рамки потом прижимаются к линиям стен
    (fp.rooms_cv.snap_room_bbox). Возвращает ([{"name","named","bbox"}] в пикселях картинки, стоимость $)."""
    import anthropic
    h, w = img.shape[:2]
    x0 = y0 = 0
    x1, y1 = w, h
    if region and len(region) == 4:
        px, py = 0.03 * w, 0.03 * h
        x0, y0 = int(max(0, region[0] - px)), int(max(0, region[1] - py))
        x1, y1 = int(min(w, region[2] + px)), int(min(h, region[3] + py))
    data, k = _png(img[y0:y1, x0:x1], FULL_SIDE)
    tw, th = round((x1 - x0) * k), round((y1 - y0) * k)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
               {"type": "text", "text": (
                   f"This is a building plan drawing ({tw}x{th} px). List every ROOM of the building: each space "
                   "fully enclosed by walls (including mechanical / pump rooms, electrical rooms, toilets, storage, "
                   "corridors). A room full of pipes or equipment is still ONE room - its boundary is the walls, "
                   "not the pipes. Do NOT list: pipes, tanks, pits, equipment, outdoor pads or slabs outside the "
                   "walls, detail callouts, notes, the title block, the building's outer wall band or column grid "
                   "strip. List every room SEPARATELY, even if many rooms are identical: never one box "
                   "for a group of repeated rooms and never the word 'typical'. For each room: the name written in it (named_on_sheet=true), or a short description "
                   "such as 'pump room' (named_on_sheet=false); bbox of the room's inner wall faces in this "
                   "image's pixels. If there are no walls at all, return an empty list.")}]
    resp = anthropic.Anthropic().messages.create(
        model=VLM_MODEL, max_tokens=3000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": ROOMS_SCHEMA}},
        messages=[{"role": "user", "content": content}])
    pin, pout = PRICES.get(VLM_MODEL, (4.0, 20.0))
    cost = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
    if resp.stop_reason == "refusal":
        return [], cost
    r = json.loads(next(b.text for b in resp.content if b.type == "text"))
    out = []
    for rm in r["rooms"]:
        b = rm["bbox"]
        if len(b) != 4 or b[2] <= b[0] or b[3] <= b[1]:
            continue
        out.append({"name": rm["name"], "named": rm["named_on_sheet"],
                    "bbox": [x0 + b[0] / k, y0 + b[1] / k, x0 + b[2] / k, y0 + b[3] / k]})
    return out, cost


# ---------------- лист трубопроводов: подписи диаметров и марки ----------------
PIPING_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {
        "text": {"type": "string", "description": "the label exactly as written, e.g. 8\", 16\" SUCTION HEADER, CV-3, PG"},
        "kind": {"type": "string", "enum": ["pipe_size", "valve", "instrument", "support", "equipment", "other"]},
        "size_in": {"type": "number", "description": "pipe size in inches for pipe_size labels, else 0"},
        "x": _num, "y": _num,
        "target_x": {"type": "number", "description": "where the label's leader/arrow points (the pipe or item); "
                                                       "the label centre if there is no leader"},
        "target_y": _num},
    "required": ["text", "kind", "size_in", "x", "y", "target_x", "target_y"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False}

DUP_PX = 35   # ~0.5 м на M-101A: одна подпись из двух проходов чтения ложится с разбросом до ~30 px

PIPING_PROMPT = """This is one tile of a mechanical PIPING plan ({tw}x{th} px). List EVERY text label in it that belongs \
to the piping: pipe size callouts (8", 6", 2", 1", 16" SUCTION HEADER, 12" PUMP DISCHARGE HEADER, 10"x12" REDUCER...), \
valve tags (CV-1, FSCV-3, BPCV, MAV, ...), instrument tags (PG, PS = pressure switch only when written in a circle, \
DPT-1, PIT 3, FS, ...), pipe support tags (PS-1, PS-12 with a leader to a dot on the pipe), equipment tags (FP-1, FSI-2, \
FSR-1, pumps, tanks). Skip general notes, title block text, room names and sheet references (M-303, A1). \
For each: the text exactly as written, its kind, size_in for pipe sizes (else 0), the label centre (x, y) and the \
point its leader arrow touches (target_x, target_y) in this tile's pixels. A label that applies to several identical \
items (TYP.) is listed once."""


def read_piping(img: np.ndarray, region=None, model: str | None = None, passes: int = 2) -> tuple[list[dict], float]:
    """Подписи трубопроводов по увеличенным фрагментам (3×2) рамки чертежа: диаметры с точкой выноски,
    марки арматуры, приборов, опор и оборудования. Координаты — в пикселях исходной картинки.
    passes — сколько раз читать каждый фрагмент (параллельно); найденное объединяется: за один проход модель
    каждый раз пропускает разные подписи, и итоги между загрузками прыгают."""
    import anthropic
    model = model or VLM_MODEL
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in (region or (0, 0, w, h)))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    nx, ny, ov = 3, 2, 40
    xs = np.linspace(x0, x1, nx + 1).astype(int)
    ys = np.linspace(y0, y1, ny + 1).astype(int)
    client = anthropic.Anthropic()
    pin, pout = PRICES.get(model, (4.0, 20.0))

    def tile(ij):
        j, i = ij
        tx0, ty0 = max(x0, xs[i] - ov), max(y0, ys[j] - ov)
        tx1, ty1 = min(x1, xs[i + 1] + ov), min(y1, ys[j + 1] + ov)
        data, k = _png(cv2.resize(img[ty0:ty1, tx0:tx1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                       TILE_SIDE)
        k *= 2
        tw, th = round((tx1 - tx0) * k), round((ty1 - ty0) * k)
        resp = client.messages.create(
            model=model, max_tokens=8000,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": PIPING_SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                {"type": "text", "text": PIPING_PROMPT.format(tw=tw, th=th)}]}])
        c = (resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6
        text = next((b.text for b in resp.content if b.type == "text"), None) \
            if resp.stop_reason != "refusal" else None
        got = []
        for it in (json.loads(text)["items"] if text else []):
            for a, b in (("x", "y"), ("target_x", "target_y")):
                it[a] = round(tx0 + it[a] / k, 1)
                it[b] = round(ty0 + it[b] / k, 1)
            it["tile"] = j * nx + i
            got.append(it)
        return got, c

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(nx * ny * passes) as ex:
        res = list(ex.map(tile, [(j, i) for _ in range(passes) for j in range(ny) for i in range(nx)]))
    items = [it for got, _ in res for it in got]
    cost = sum(c for _, c in res)
    # дубли (перекрытие фрагментов, повторные проходы): тот же текст, подпись или точка выноски ближе DUP_PX
    out = []
    for it in items:
        if not any(o["text"] == it["text"] and (
                (abs(o["x"] - it["x"]) < DUP_PX and abs(o["y"] - it["y"]) < DUP_PX) or
                (abs(o["target_x"] - it["target_x"]) < DUP_PX and abs(o["target_y"] - it["target_y"]) < DUP_PX))
                for o in out):
            out.append(it)
    return out, cost
