"""IFC -> elements. Свой разбор STEP, без ifcopenshell.

Колёса ifcopenshell собраны под glibc новее, чем в этой машине, поэтому
читаем формат сами. Нам нужны только типы сущностей, привязка к этажам и
наборы количеств — это обычный текст ISO-10303-21:

    #123=IFCDOOR('2O2Fr$t4X7Zf8NOew3FNrX',#5,'Door-01',$,...);

Проверено на превью (сверено с их ответами):
  стены   = IFCWALL + IFCWALLSTANDARDCASE   (99 + 747 = 846)
  сантех. = IFCFLOWTERMINAL                  (119; IFCSANITARYTERMINAL даёт 0)
  помещения привязаны к этажу через IFCRELAGGREGATES, а двери и стены —
  через IFCRELCONTAINEDINSPATIALSTRUCTURE. Нужны обе связи.
"""
import json
import re

from labels import norm, subtype_for

# тип вопроса -> классы IFC
TYPES = {
    "door":      {"IFCDOOR"},
    "window":    {"IFCWINDOW"},
    "room":      {"IFCSPACE"},
    "wall":      {"IFCWALL", "IFCWALLSTANDARDCASE"},
    "column":    {"IFCCOLUMN"},
    "beam":      {"IFCBEAM"},
    "slab":      {"IFCSLAB"},
    "stair":     {"IFCSTAIR"},
    "railing":   {"IFCRAILING"},
    "fixture":   {"IFCFLOWTERMINAL", "IFCSANITARYTERMINAL"},
    "furniture": {"IFCFURNISHINGELEMENT"},
    "covering":  {"IFCCOVERING"},
}
CLASS2TYPE = {c: t for t, cs in TYPES.items() for c in cs}

REL = {"IFCRELCONTAINEDINSPATIALSTRUCTURE", "IFCRELAGGREGATES",
       "IFCRELDEFINESBYPROPERTIES"}
GEOM = {"IFCPRODUCTDEFINITIONSHAPE", "IFCSHAPEREPRESENTATION", "IFCPOLYLINE",
        "IFCCARTESIANPOINT", "IFCGEOMETRICCURVESET", "IFCSIUNIT"}
QTY = {"IFCELEMENTQUANTITY", "IFCQUANTITYAREA", "IFCQUANTITYVOLUME",
       "IFCQUANTITYLENGTH"}
KEEP = set(CLASS2TYPE) | REL | QTY | GEOM | {"IFCBUILDINGSTOREY"}

HEAD = re.compile(r"^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(", re.I)
# \X2\00D6\X0\ -> Ö ; \S\A ->ä и прочее оставляем как есть
UNI = re.compile(r"\\X2\\((?:[0-9A-Fa-f]{4})+)\\X0\\")
UNI1 = re.compile(r"\\X\\([0-9A-Fa-f]{2})")


def _unescape(s: str) -> str:
    s = UNI.sub(lambda m: "".join(
        chr(int(m.group(1)[i:i + 4], 16)) for i in range(0, len(m.group(1)), 4)), s)
    s = UNI1.sub(lambda m: chr(int(m.group(1), 16)), s)
    return s.replace("''", "'")


def split_args(body: str) -> list[str]:
    """Разбивает список аргументов по запятым верхнего уровня."""
    out, depth, cur, q = [], 0, [], False
    i = 0
    while i < len(body):
        ch = body[i]
        if q:
            if ch == "'":
                if i + 1 < len(body) and body[i + 1] == "'":
                    cur.append("''"); i += 2; continue
                q = False
            cur.append(ch)
        elif ch == "'":
            q = True; cur.append(ch)
        elif ch == "(":
            depth += 1; cur.append(ch)
        elif ch == ")":
            depth -= 1; cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur).strip())
    return out


def _str(a: str):
    a = a.strip()
    if a.startswith("'") and a.endswith("'"):
        return _unescape(a[1:-1])
    return None


def _num(a: str):
    try:
        return float(a)
    except (TypeError, ValueError):
        return None


def _refs(a: str) -> list[int]:
    return [int(x) for x in re.findall(r"#(\d+)", a or "")]


def read(path: str):
    """-> (ents, counts). ents: {id: (TYPE, [args])} только для нужных типов."""
    ents, counts = {}, {}
    buf = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        in_data = False
        for line in f:
            t = line.strip()
            if not in_data:
                if t.upper().startswith("DATA"):
                    in_data = True
                continue
            if t.upper().startswith("ENDSEC"):
                break
            buf += t
            if not buf.endswith(";"):
                continue                      # сущность перенесена на строки
            stmt, buf = buf[:-1], ""
            m = HEAD.match(stmt)
            if not m:
                continue
            eid, cls = int(m.group(1)), m.group(2).upper()
            counts[cls] = counts.get(cls, 0) + 1
            if cls in KEEP:
                ents[eid] = (cls, split_args(stmt[m.end():-1]))
    return ents, counts


NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _unit_scale(ents) -> float:
    """Масштаб длины файла в метры. Площади у них бывают в мм²."""
    for cls, a in ents.values():
        if cls != "IFCSIUNIT":
            continue
        raw = ",".join(a).upper()
        if ".LENGTHUNIT." not in raw:
            continue
        if ".MILLI." in raw:
            return 0.001
        if ".CENTI." in raw:
            return 0.01
        return 1.0
    return 1.0


def _points(ents, pid, seen=None):
    """Точки контура: спускаемся по ссылкам до IfcCartesianPoint."""
    seen = seen if seen is not None else set()
    if pid in seen or pid not in ents:
        return []
    seen.add(pid)
    cls, a = ents[pid]
    if cls == "IFCCARTESIANPOINT":
        n = NUM.findall(",".join(a))
        return [(float(n[0]), float(n[1]))] if len(n) >= 2 else []
    out = []
    for r in _refs(",".join(a)):
        out += _points(ents, r, seen)
    return out


def _shoelace(p) -> float:
    n = len(p)
    if n < 3:
        return 0.0
    return abs(sum(p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1]
                   for i in range(n))) / 2


def _footprint_area(ents, eid, scale) -> float | None:
    """Площадь помещения по контуру FootPrint.

    ПРОВЕРЕНО: 200 из 200 помещений esplanades совпали с их ответами.
    Нужно потому, что у части моделей наборов количеств нет вовсе, а
    подписанная площадь округлена («5,8» против настоящих 5,76).
    """
    cls, a = ents.get(eid, (None, []))
    if cls is None:
        return None
    for r in _refs(",".join(a)):
        if r not in ents or ents[r][0] != "IFCPRODUCTDEFINITIONSHAPE":
            continue
        for rep in _refs(",".join(ents[r][1])):
            if rep in ents and ents[rep][0] == "IFCSHAPEREPRESENTATION" \
                    and "FootPrint" in ",".join(ents[rep][1]):
                area = _shoelace(_points(ents, rep)) * scale * scale
                if area > 0:
                    return area
    return None


def parse(path: str, dict_rows: dict):
    ents, counts = read(path)
    scale = _unit_scale(ents)

    # --- этажи -----------------------------------------------------------
    storeys = {}
    for eid, (cls, a) in ents.items():
        if cls != "IFCBUILDINGSTOREY":
            continue
        name = (_str(a[2]) if len(a) > 2 else None) or \
               (_str(a[7]) if len(a) > 7 else None) or ""
        lvl = _num(a[9]) if len(a) > 9 else None
        storeys[eid] = {"name": name, "level": lvl or 0.0, "guid": _str(a[0])}
    order = sorted(storeys, key=lambda k: storeys[k]["level"])
    for i, k in enumerate(order):
        storeys[k]["ord"] = i

    # --- привязка элементов к этажам -------------------------------------
    # двери, стены, окна: ContainedInSpatialStructure
    # помещения: Aggregates (этаж СОСТОИТ из помещений)
    owner: dict[int, int] = {}
    for cls, a in ents.values():
        if cls == "IFCRELCONTAINEDINSPATIALSTRUCTURE" and len(a) > 5:
            st = _refs(a[5])
            if st and st[0] in storeys:
                for e in _refs(a[4]):
                    owner[e] = st[0]
        elif cls == "IFCRELAGGREGATES" and len(a) > 5:
            st = _refs(a[4])
            if st and st[0] in storeys:
                for e in _refs(a[5]):
                    owner.setdefault(e, st[0])

    # --- количества (Qto) -------------------------------------------------
    # IfcRelDefinesByProperties -> IfcElementQuantity -> IfcQuantityArea/...
    qvals: dict[int, dict] = {}
    for cls, a in ents.values():
        if cls != "IFCRELDEFINESBYPROPERTIES" or len(a) < 6:
            continue
        pd = _refs(a[5])
        if not pd or pd[0] not in ents:
            continue
        pcls, pa = ents[pd[0]]
        if pcls != "IFCELEMENTQUANTITY" or len(pa) < 6:
            continue
        vals = {}
        for q in _refs(pa[5]):
            if q not in ents:
                continue
            qcls, qa = ents[q]
            if qcls not in ("IFCQUANTITYAREA", "IFCQUANTITYVOLUME",
                            "IFCQUANTITYLENGTH") or len(qa) < 4:
                continue
            nm, v = _str(qa[0]), _num(qa[3])
            if nm and v is not None:
                vals[nm] = v
        if vals:
            for e in _refs(a[4]):
                qvals.setdefault(e, {}).update(vals)

    AREA = ("NetFloorArea", "GrossFloorArea", "NetArea", "GrossArea", "Area")
    VOL = ("NetVolume", "GrossVolume", "Volume")
    LEN = ("Length", "NetLength")

    def pick(e, keys):
        v = qvals.get(e, {})
        for k in keys:
            if k in v:
                return v[k]
        return None

    # --- элементы ---------------------------------------------------------
    out = []
    for eid, (cls, a) in ents.items():
        t = CLASS2TYPE.get(cls)
        if not t:
            continue
        guid = _str(a[0]) if a else None
        name = _str(a[2]) if len(a) > 2 else None
        longname = _str(a[7]) if len(a) > 7 else None

        # у помещения два имени: Name — номер («4-3»), LongName — название
        # («TUALETTRUUM»). В вопросах они склеены: «TUALETTRUUM (34-3)».
        if t == "room" and longname and name:
            label = f"{longname} ({name})"
        else:
            label = longname or name

        st = owner.get(eid)
        row = {
            "type": t,
            "subtype": subtype_for(longname or name or "", dict_rows) if t == "room" else None,
            "label": label,
            "label_norm": norm(longname or name or "") if label else None,
            "source_ref": guid,
            "floor_name": storeys[st]["name"] if st in storeys else None,
            "props": json.dumps({"ifc_class": cls, "name": name,
                                 "long_name": longname}, ensure_ascii=False),
        }
        if t == "room":
            # сначала посчитанное количество, иначе считаем контур сами
            row["area"] = pick(eid, AREA) or _footprint_area(ents, eid, scale)
            row["volume"] = pick(eid, VOL)
        elif t in ("wall", "beam"):
            row["length"] = pick(eid, LEN)
        elif t == "slab":
            row["area"] = pick(eid, AREA)
            row["volume"] = pick(eid, VOL)
        out.append(row)

    fl = [{"name": storeys[k]["name"], "level": storeys[k]["level"],
           "ord": storeys[k]["ord"]} for k in order]
    return fl, out
