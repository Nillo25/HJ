"""DXF swiss_dwellings -> elements.

Слои: ROOM_OUTLINE, ROOM_TEXT, UNIT_OUTLINE, WALL, DOOR, WINDOW, FIXTURE, ANNOTATION
Координаты уже в метрах («Masse in Metern»), площадь полигона выходит в м².

Парсер свой, без ezdxf: ASCII DXF — это чередующиеся строки «код / значение».
Проверено на site1000_bldg2082_floor9385: квартиры 4=4, двери 36=36, окна 28=28,
площадь комнаты 16.78=16.78.
"""
import json
from labels import classify_text, norm, subtype_for


def read_entities(path: str) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = [l.rstrip("\n") for l in f]
    pairs = [(raw[i].strip(), raw[i + 1].strip()) for i in range(0, len(raw) - 1, 2)]
    start = end = None
    for i, (c, v) in enumerate(pairs):
        if c == "2" and v == "ENTITIES":
            start = i + 1
        if start is not None and c == "0" and v == "ENDSEC":
            end = i
            break
    if start is None:
        return []
    ents, cur = [], None
    for c, v in pairs[start:end]:
        if c == "0":
            cur = {"type": v}
            ents.append(cur)
        elif cur is not None:
            cur.setdefault(c, []).append(v)
    return ents


def layer(e) -> str:
    return (e.get("8") or [""])[0]


def pts(e) -> list[tuple[float, float]]:
    xs = [float(x) for x in e.get("10", [])]
    ys = [float(y) for y in e.get("20", [])]
    return list(zip(xs, ys))[: min(len(xs), len(ys))]


def shoelace(p) -> float:
    n = len(p)
    if n < 3:
        return 0.0
    s = sum(p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1] for i in range(n))
    return abs(s) / 2


def bbox(p):
    if not p:
        return None
    xs, ys = [a for a, _ in p], [b for _, b in p]
    return [min(xs), min(ys), max(xs), max(ys)]


def inside(pt, poly) -> bool:
    """Точка в многоугольнике, луч вправо."""
    x, y = pt
    c = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            c = not c
    return c


def centroid(p):
    return (sum(a for a, _ in p) / len(p), sum(b for _, b in p) / len(p))


def parse(path: str, dict_rows: dict) -> list[dict]:
    """Возвращает список строк для таблицы elements (без drawing_id/file_id)."""
    ents = read_entities(path)
    out: list[dict] = []

    # --- комнаты -------------------------------------------------------
    rooms = []
    for e in ents:
        if e["type"] == "LWPOLYLINE" and layer(e) == "ROOM_OUTLINE":
            p = pts(e)
            if len(p) >= 3:
                rooms.append({"polygon": p, "area": shoelace(p),
                              "handle": (e.get("5") or [""])[0],
                              "name": None, "number": None, "printed_area": None})

    # --- подписи: раскидываем по комнатам, в которые попали -------------
    texts = []
    for e in ents:
        if e["type"] in ("TEXT", "MTEXT"):
            s = (e.get("1") or [""])[0]
            p = pts(e)
            if not s or not p:
                continue
            texts.append({"text": s, "at": p[0], "layer": layer(e),
                          "kind": classify_text(s)})
            out.append({"type": "text_span", "label": s, "label_norm": norm(s),
                        "bbox": json.dumps([p[0][0], p[0][1], p[0][0], p[0][1]]),
                        "props": json.dumps({"layer": layer(e), "kind": classify_text(s)})})

    for t in texts:
        if t["layer"] != "ROOM_TEXT":
            continue
        for r in rooms:
            if inside(t["at"], r["polygon"]):
                if t["kind"] == "name" and not r["name"]:
                    r["name"] = t["text"]
                elif t["kind"] == "number" and not r["number"]:
                    r["number"] = t["text"]
                elif t["kind"] == "area" and not r["printed_area"]:
                    r["printed_area"] = t["text"]
                break

    for r in rooms:
        lbl = r["name"] or r["number"]
        out.append({
            "type": "room",
            "subtype": subtype_for(r["name"], dict_rows) if r["name"] else None,
            "label": lbl,
            "label_norm": norm(lbl) if lbl else None,
            "area": round(r["area"], 4),
            "polygon": json.dumps([[round(x, 4), round(y, 4)] for x, y in r["polygon"]]),
            "bbox": json.dumps(bbox(r["polygon"])),
            "source_ref": r["handle"],
            "props": json.dumps({
                "number": r["number"],
                "printed_area": r["printed_area"],
                # пронумерованное внутреннее помещение: есть номер, и это не балкон
                "numbered_indoor": bool(r["number"]) and
                                   subtype_for(r["name"] or "", dict_rows) != "balcony",
            }),
        })

    # --- квартиры (UNIT_OUTLINE) ---------------------------------------
    units = [pts(e) for e in ents
             if e["type"] == "LWPOLYLINE" and layer(e) == "UNIT_OUTLINE"]
    for i, p in enumerate(units):
        if len(p) < 3:
            continue
        # площадь квартиры = сумма комнат внутри контура (НЕ площадь контура:
        # контур включает стены и даёт завышение ~2-3%)
        inner = [r for r in rooms if inside(centroid(r["polygon"]), p)]
        out.append({
            "type": "apartment",
            "label": chr(ord("A") + i),
            "label_norm": chr(ord("a") + i),
            "area": round(sum(r["area"] for r in inner), 4),
            "polygon": json.dumps([[round(x, 4), round(y, 4)] for x, y in p]),
            "bbox": json.dumps(bbox(p)),
            "props": json.dumps({"outline_area": round(shoelace(p), 4),
                                 "rooms": len(inner)}),
        })

    # --- окна: ПРОВЕРЕНО 100% на 65 файлах — три полилинии на одно окно --
    wins = [pts(e) for e in ents
            if e["type"] == "LWPOLYLINE" and layer(e) == "WINDOW"]
    for i in range(0, len(wins) - 2, 3):
        group = wins[i] + wins[i + 1] + wins[i + 2]
        out.append({"type": "window", "bbox": json.dumps(bbox(group))})

    # --- двери: СЛАБОЕ МЕСТО, 56% на 65 файлах --------------------------
    # дуга есть только у распашной двери; раздвижные её не имеют,
    # двустворчатые дают две. Чинить здесь.
    arcs = [e for e in ents if e["type"] == "ARC" and layer(e) == "DOOR"]
    for a in arcs:
        cx = float((a.get("10") or [0])[0])
        cy = float((a.get("20") or [0])[0])
        r = float((a.get("40") or [0])[0])
        out.append({"type": "door", "subtype": "single",
                    "bbox": json.dumps([cx - r, cy - r, cx + r, cy + r]),
                    "source_ref": (a.get("5") or [""])[0],
                    "props": json.dumps({"rule": "arc_on_door_layer", "weak": True})})

    # --- стены ----------------------------------------------------------
    for e in ents:
        if e["type"] == "LWPOLYLINE" and layer(e) == "WALL":
            p = pts(e)
            if p:
                out.append({"type": "wall", "polygon": json.dumps(p),
                            "bbox": json.dumps(bbox(p))})

    # --- сантехника: НЕ РАЗЛИЧЕНА по типам ------------------------------
    # всё на слое FIXTURE: полилинии и окружности. Различать по форме/размеру.
    for e in ents:
        if layer(e) != "FIXTURE":
            continue
        if e["type"] == "LWPOLYLINE":
            p = pts(e)
            if p:
                out.append({"type": "fixture", "subtype": None,
                            "bbox": json.dumps(bbox(p)),
                            "props": json.dumps({"shape": "poly",
                                                 "area": round(shoelace(p), 4)})})
        elif e["type"] == "CIRCLE":
            cx = float((e.get("10") or [0])[0])
            cy = float((e.get("20") or [0])[0])
            r = float((e.get("40") or [0])[0])
            out.append({"type": "fixture", "subtype": None,
                        "bbox": json.dumps([cx - r, cy - r, cx + r, cy + r]),
                        "props": json.dumps({"shape": "circle", "r": round(r, 4)})})

    return out
