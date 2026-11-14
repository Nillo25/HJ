"""Функции ответа поверх базы + сопоставление известных шаблонов вопроса.

Модель здесь не участвует: это быстрый путь (кэш шаблонов) и те же функции,
которые потом будет вызывать LLM через tools. Благодаря этому стенд оценки
работает ещё до того, как появился разбор вопроса моделью.

Любая функция возвращает И число, И список элементов — из него собирается
evidence. Функция, вернувшая число без элементов, считается ошибкой.
"""
import json
import re

# ---------------------------------------------------------------- функции

def _rows(con, drawing, **w):
    q = ["SELECT * FROM elements WHERE drawing_id=(SELECT id FROM drawings WHERE name=?)"]
    args = [drawing]
    for k, v in w.items():
        if v is None:
            continue
        if k == "floor":
            q.append("AND floor_id=(SELECT id FROM floors WHERE name=? AND "
                     "drawing_id=(SELECT id FROM drawings WHERE name=?))")
            args += [v, drawing]
        elif k == "label_literal":
            q.append("AND LOWER(label) LIKE ?")
            args.append(f"%{v.lower()}%")
        else:
            q.append(f"AND {k}=?")
            args.append(v)
    con.row_factory = __import__("sqlite3").Row
    return con.execute(" ".join(q), args).fetchall()


def count_objects(con, drawing, type, subtype=None, floor=None, label_literal=None):
    r = _rows(con, drawing, type=type, subtype=subtype, floor=floor,
              label_literal=label_literal)
    if not r and subtype:
        # есть ли вообще объекты этого типа с проставленным подтипом?
        # если ни у одного подтип не определён — честный отказ, а не ноль.
        any_typed = [x for x in _rows(con, drawing, type=type) if x["subtype"]]
        if not any_typed:
            return None, []
    return len(r), r


def room_area(con, drawing, label, floor=None):
    lab = label.strip().lower()
    all_rooms = _rows(con, drawing, type="room", floor=floor)
    r = [x for x in all_rooms if x["label"] and lab in x["label"].lower()]
    if not r:   # спрашивают по номеру: «room 1.02 ('Stanza')»
        r = [x for x in all_rooms
             if (json.loads(x["props"] or "{}").get("number") or "").lower() == lab]
    if not r:
        return None, []
    return sum(x["area"] or 0 for x in r), r


def apartment_area(con, drawing, label, floor=None):
    """Площадь квартиры по букве с плана («Whg. B» -> B)."""
    r = [x for x in _rows(con, drawing, type="apartment")
         if (x["label"] or "").upper() == label.upper()]
    if not r:
        return None, []
    return sum(x["area"] or 0 for x in r), r


def count_labelled(con, drawing, literal, floor=None):
    """«Сколько помещений с подписью 'Cucina'» — считаем сами подписи на
    слое ROOM_TEXT, а не комнаты: подпись может не попасть внутрь контура,
    и тогда комната её теряет, а вопрос всё равно про подпись."""
    did = con.execute("SELECT id FROM drawings WHERE name=?", (drawing,)).fetchone()
    if not did:
        return None, []
    con.row_factory = __import__("sqlite3").Row
    rows = con.execute(
        """SELECT t.* FROM text_spans t JOIN files f ON f.id = t.file_id
            WHERE f.drawing_id = ? AND LOWER(TRIM(t.text)) = LOWER(?)""",
        (did[0], literal)).fetchall()
    return len(rows), rows


def sum_quantity(con, drawing, type, field, subtype=None, floor=None,
                 only_numbered=False, label_literal=None):
    r = _rows(con, drawing, type=type, subtype=subtype, floor=floor,
              label_literal=label_literal)
    if only_numbered:
        r = [x for x in r
             if (json.loads(x["props"] or "{}")).get("numbered_indoor")]
    if not r:
        return None, []
    return sum(x[field] or 0 for x in r), r


def list_floors(con, drawing):
    con.row_factory = __import__("sqlite3").Row
    r = con.execute("SELECT * FROM floors WHERE drawing_id="
                    "(SELECT id FROM drawings WHERE name=?) ORDER BY ord",
                    (drawing,)).fetchall()
    return len(r), r


# ------------------------------------------------- шаблоны известных вопросов
# (регексп, функция, аргументы). Покрывают датасет; живой вопрос судьи
# сюда не попадёт и уйдёт в LLM-путь.

T = [
    # --- swiss_dwellings + cubicasa5k: подсчёты ---
    (r"how many apartments|wie viele wohnungen|combien d.appartements",
     "count_objects", {"type": "apartment"}),
    (r"how many doors|wie viele t[uü]ren|kuinka monta ovea",
     "count_objects", {"type": "door"}),
    (r"how many windows|wie viele fenster|kuinka monta ikkunaa",
     "count_objects", {"type": "window"}),
    (r"how many toilets|how many toilet bowls|kuinka monta wc-istuinta",
     "count_objects", {"type": "fixture", "subtype": "toilet"}),
    (r"how many washbasins|how many sinks|kuinka monta pesu",
     "count_objects", {"type": "fixture", "subtype": "sink"}),
    (r"how many bathtubs|how many showers and bathtubs|kuinka monta suihkua",
     "count_objects", {"type": "fixture", "subtype": "bathtub"}),
    (r"how many saunas|kuinka monta saunaa",
     "count_objects", {"type": "room", "subtype": "sauna"}),
    (r"how many kitchens|kuinka monta keitti",
     "count_objects", {"type": "room", "subtype": "kitchen"}),
    (r"how many bedrooms|kuinka monta makuuhuonetta",
     "count_objects", {"type": "room", "subtype": "bedroom"}),
    (r"how many separate wc rooms|kuinka monta erillist",
     "count_objects", {"type": "room", "subtype": "wc"}),
    (r"how many bathrooms or washrooms|kuinka monta kylpy",
     "count_objects", {"type": "room", "subtype": "bathroom"}),
    (r"how many balconies|kuinka monta parveketta",
     "count_objects", {"type": "room", "subtype": "balcony"}),
    # --- комнаты по буквальной подписи: "rooms labelled 'Cucina'" ---
    (r"how many rooms labelled '(?P<lit>.+)' are there", "count_labelled", {}),
    (r"how many rooms labelled \"(?P<lit>.+)\" are there", "count_labelled", {}),
    # --- площади ---
    (r"total net floor area .* all numbered indoor rooms",
     "sum_quantity", {"type": "room", "field": "area", "only_numbered": True}),
    (r"net floor area of apartment (?P<apt>[A-Z])\b",
     "apartment_area", {"_apt": True}),
    (r"net floor area of room (?P<room>[\d.\-]+)",
     "room_area", {"_room": True}),
    (r"total net area .* rooms labelled '(?P<lit>.+)' on this floor",
     "sum_quantity", {"type": "room", "field": "area"}),
    (r"total net area .* rooms labelled \"(?P<lit>.+)\" on this floor",
     "sum_quantity", {"type": "room", "field": "area"}),
    # --- ifc_models: по этажу ---
    (r"how many (?P<kind>doors|windows|walls|columns|beams|stairs|rooms \(spaces\))"
     r" are shown on the floor plan of ['\"“](?P<floor>[^'\"”]+)['\"”]",
     "count_objects", {"_ifc_floor": True}),
    (r"how many (?P<kind>doors|windows|walls|columns|beams|stairs|sanitary fixtures|"
     r"rooms \(spaces\)) are there in the model",
     "count_objects", {"_ifc_model": True}),
]

IFC_KIND = {"doors": "door", "windows": "window", "walls": "wall",
            "columns": "column", "beams": "beam", "stairs": "stair",
            "rooms (spaces)": "room", "sanitary fixtures": "fixture"}


def match_template(question: str):
    """-> (fn_name, kwargs) либо None."""
    q = question.strip().lower()
    for rx, fn, base in T:
        m = re.search(rx, q, re.I)
        if not m:
            continue
        kw = {k: v for k, v in base.items() if not k.startswith("_")}
        g = m.groupdict()
        if g.get("lit"):
            kw["literal" if fn == "count_labelled" else "label_literal"] = g["lit"]
        if base.get("_ifc_floor"):
            kw["type"] = IFC_KIND[g["kind"]]
            kw["floor"] = g["floor"]
        if base.get("_ifc_model"):
            kw["type"] = IFC_KIND[g["kind"]]
        if base.get("_apt"):
            kw["label"] = g["apt"]
        if base.get("_room"):
            kw["label"] = g["room"]
        if g.get("floor") and "floor" not in kw:
            kw["floor"] = g["floor"]
        return fn, kw
    return None


# ------------------------------------------------------------------ фасад

FNS = {"count_objects": count_objects, "room_area": room_area,
       "apartment_area": apartment_area, "count_labelled": count_labelled,
       "sum_quantity": sum_quantity, "list_floors": list_floors}


def evidence_of(rows, limit=50):
    out = []
    for r in rows[:limit]:
        k = set(r.keys())
        if "source_ref" not in k:          # text_spans: подпись на листе
            out.append({"kind": "page", "id": r["id"], "label": r["text"],
                        "bbox": r["bbox"]})
            continue
        if r["source_ref"] and len(str(r["source_ref"])) == 22:
            out.append({"kind": "ifc", "ref": r["source_ref"], "label": r["label"]})
        elif r["polygon"]:
            out.append({"kind": "geom", "ref": r["source_ref"], "id": r["id"],
                        "label": r["label"]})
        else:
            out.append({"kind": "geom", "id": r["id"], "label": r["label"]})
    return out


def ask(con, drawing: str, question: str) -> dict:
    """Ответ в форме контракта. route='cache' — без единого вызова модели."""
    m = match_template(question)
    if not m:
        return {"status": "refused", "reason": "тип вопроса не распознан",
                "missing": "template", "route": "cache", "cost_usd": 0.0}
    fn, kw = m
    try:
        value, rows = FNS[fn](con, drawing, **kw)
    except Exception as e:
        return {"status": "refused", "reason": f"{type(e).__name__}: {e}",
                "missing": "data", "route": "cache", "cost_usd": 0.0}
    if value is None or (not rows and fn != "count_objects"):
        return {"status": "refused", "reason": "в данных не найдено",
                "missing": "element", "searched": json.dumps(kw, ensure_ascii=False),
                "route": "cache", "cost_usd": 0.0}
    unit = ("m2" if fn in ("room_area", "apartment_area")
            or kw.get("field") == "area" else "count")
    conf = min([r["confidence"] for r in rows], default=1.0) if rows else 1.0
    return {"status": "answered" if conf >= 1.0 else "estimated",
            "answer": round(value, 2) if unit == "m2" else value,
            "unit": unit, "confidence": conf, "route": "cache",
            "evidence": evidence_of(rows), "cost_usd": 0.0,
            "fn": fn, "args": kw}
