"""DXF swiss_dwellings -> elements.

Слои: ROOM_OUTLINE, ROOM_TEXT, UNIT_OUTLINE, WALL, DOOR, WINDOW, FIXTURE, ANNOTATION
Координаты уже в метрах («Masse in Metern»), площадь полигона выходит в м².

Парсер свой, без ezdxf: ASCII DXF — это чередующиеся строки «код / значение».
Проверено на site1000_bldg2082_floor9385: квартиры 4=4, двери 36=36, окна 28=28,
площадь комнаты 16.78=16.78.
"""
import json
import re as _re

import group as G
from labels import classify_text, is_outdoor, norm, subtype_for

# «Whg. B», «App. A», «Wohnung C», «Appartamento D» — буква квартиры
# написана на самом плане. Это надёжнее порядка обхода файла.
APT_RE = _re.compile(
    r"^\s*(?:whg|wohnung|app|appartamento|appartement|apt|apartment)\.?\s*"
    r"([A-Z])\b", _re.I)


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
                "outdoor": is_outdoor(subtype_for(r["name"] or "", dict_rows)),
                "labeled": bool(lbl),
                # «все пронумерованные внутренние помещения, кроме балконов,
                # лоджий, террас, шахт и пустот»: есть подпись, есть номер,
                # не наружное. Безымянные обрезки (<1 м², без подписи) — это
                # и есть шахты и пустоты, они отсекаются условием labeled.
                "numbered_indoor": bool(lbl) and bool(r["number"]) and
                                   not is_outdoor(subtype_for(r["name"] or "", dict_rows)),
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

        # буква квартиры — с самого плана, если она там написана
        letter = None
        for r in inner:
            m = APT_RE.match(r["name"] or "")
            if m:
                letter = m.group(1).upper()
                break
        if letter is None:
            letter = chr(ord("A") + i)

        # в площадь квартиры входят только внутренние помещения с подписью
        counted = [r for r in inner
                   if (r["name"] or r["number"])
                   and not is_outdoor(subtype_for(r["name"] or "", dict_rows))]

        out.append({
            "type": "apartment",
            "label": letter,
            "label_norm": letter.lower(),
            "area": round(sum(r["area"] for r in counted), 4),
            "polygon": json.dumps([[round(x, 4), round(y, 4)] for x, y in p]),
            "bbox": json.dumps(bbox(p)),
            "props": json.dumps({"outline_area": round(shoelace(p), 4),
                                 "rooms": len(counted),
                                 "rooms_inside": len(inner),
                                 "letter_from_plan": letter != chr(ord("A") + i)}),
        })

    # --- окна: ПРОВЕРЕНО 100% на 65 файлах — три полилинии на одно окно --
    wins = [pts(e) for e in ents
            if e["type"] == "LWPOLYLINE" and layer(e) == "WINDOW"]
    for i in range(0, len(wins) - 2, 3):
        group = wins[i] + wins[i + 1] + wins[i + 2]
        out.append({"type": "window", "bbox": json.dumps(bbox(group))})

    # --- двери: полилинии минус дуги -------------------------------------
    # ПРОВЕРЕНО: 65 из 65 файлов. У каждой двери своя полилиния-полотно,
    # и каждая дуга раскрытия добавляет ещё одну. Значит дверей ровно
    # (полилиний - дуг). Наивный счёт дуг давал 56%, склейка по близости
    # слепляла соседние двери и давала недосчёт.
    door_polys = [pts(e) for e in ents
                  if e["type"] == "LWPOLYLINE" and layer(e) == "DOOR"]
    door_polys = [p for p in door_polys if p]
    door_arcs = []
    for e in ents:
        if e["type"] == "ARC" and layer(e) == "DOOR":
            cx = float((e.get("10") or [0])[0]); cy = float((e.get("20") or [0])[0])
            r = float((e.get("40") or [0])[0])
            door_arcs.append((cx, cy, r))

    # каждой дуге сопоставляем ближайшую свободную полилинию и убираем её:
    # останется ровно (полилиний - дуг) штук, и у каждой есть геометрия
    taken = set()
    for cx, cy, r in door_arcs:
        best, bestd = None, None
        for i, p in enumerate(door_polys):
            if i in taken:
                continue
            c = centroid(p)
            d = (c[0] - cx) ** 2 + (c[1] - cy) ** 2
            if bestd is None or d < bestd:
                best, bestd = i, d
        if best is not None:
            taken.add(best)

    for i, p in enumerate(door_polys):
        if i in taken:
            continue
        out.append({"type": "door", "subtype": "single",
                    "bbox": json.dumps(bbox(p)),
                    "polygon": json.dumps([[round(x, 4), round(y, 4)] for x, y in p]),
                    "props": json.dumps({"rule": "polys_minus_arcs"})})

    # --- стены ----------------------------------------------------------
    for e in ents:
        if e["type"] == "LWPOLYLINE" and layer(e) == "WALL":
            p = pts(e)
            if p:
                out.append({"type": "wall", "polygon": json.dumps(p),
                            "bbox": json.dumps(bbox(p))})

    # --- сантехника: склейка + габарит + комната --------------------------
    fx_prims = []
    for e in ents:
        if layer(e) != "FIXTURE":
            continue
        if e["type"] == "LWPOLYLINE":
            p = pts(e)
            if p:
                fx_prims.append({"bbox": bbox(p), "shape": "poly"})
        elif e["type"] == "CIRCLE":
            cx = float((e.get("10") or [0])[0]); cy = float((e.get("20") or [0])[0])
            r = float((e.get("40") or [0])[0])
            fx_prims.append({"bbox": [cx - r, cy - r, cx + r, cy + r], "shape": "circle"})
        elif e["type"] == "ARC":
            cx = float((e.get("10") or [0])[0]); cy = float((e.get("20") or [0])[0])
            r = float((e.get("40") or [0])[0])
            fx_prims.append({"bbox": [cx - r, cy - r, cx + r, cy + r], "shape": "arc"})

    for g in G.cluster(fx_prims, gap=0.03):
        bb = g["bbox"]
        c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
        room_sub = None
        for r in rooms:
            if inside(c, r["polygon"]):
                room_sub = subtype_for(r["name"] or "", dict_rows)
                break
        sub, conf = G.classify_fixture(g, room_sub)
        w, h, long, short, aspect = G.dims(bb)
        out.append({"type": "fixture", "subtype": sub, "confidence": conf,
                    "bbox": json.dumps(bb),
                    "props": json.dumps({"room": room_sub, "prims": g["n"],
                                         "long": round(long, 3),
                                         "short": round(short, 3),
                                         "aspect": round(aspect, 2)})})

    return out
