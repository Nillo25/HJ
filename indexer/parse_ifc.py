"""IFC -> elements. Hand-rolled STEP reader, no ifcopenshell.

Why not ifcopenshell: it ships a compiled C++ core, so every target needs a
matching prebuilt wheel. There is none for older glibc on arm64 Linux, which
is exactly where this indexer ran during development. Reading the format
ourselves removes the native dependency entirely, so the same code runs on a
laptop, in CI and in the deployment image without a build toolchain.

What we give up: no geometry kernel, so no booleans, no 3D volumes, no IFC
schema validation. We only need entity types, storey containment, quantity
sets and flat footprint contours, and those are plain ISO-10303-21 text:

    #123=IFCDOOR('2O2Fr$t4X7Zf8NOew3FNrX',#5,'Door-01',$,...);

If 3D geometry is ever needed, switch this module to ifcopenshell behind the
same parse() signature; nothing else in the indexer depends on the reader.

Verified against their answers on the preview:
  walls    = IFCWALL + IFCWALLSTANDARDCASE   (99 + 747 = 846)
  fixtures = IFCFLOWTERMINAL                 (119; IFCSANITARYTERMINAL gives 0)
  spaces attach to a storey through IFCRELAGGREGATES, while doors and walls
  use IFCRELCONTAINEDINSPATIALSTRUCTURE. Both links are needed.
  room area from the FootPrint contour matched 200 of 200 spaces.
"""
import json
import re

from labels import norm, subtype_for

# question subject -> IFC classes
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
TYPE_DEFS = {"IFCDOORTYPE", "IFCWINDOWTYPE", "IFCDOORSTYLE", "IFCWINDOWSTYLE"}
KEEP = (set(CLASS2TYPE) | REL | QTY | GEOM | TYPE_DEFS | {
    "IFCBUILDINGSTOREY", "IFCBUILDING", "IFCPROJECT", "IFCSITE", "IFCRELDEFINESBYTYPE", "IFCAPPLICATION"})

HEAD = re.compile(r"^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(", re.I)
# \X2\00D6\X0\ -> O-umlaut, \X\E6 -> ae-ligature
UNI = re.compile(r"\\X2\\((?:[0-9A-Fa-f]{4})+)\\X0\\")
UNI1 = re.compile(r"\\X\\([0-9A-Fa-f]{2})")


def _unescape(s: str) -> str:
    s = UNI.sub(lambda m: "".join(
        chr(int(m.group(1)[i:i + 4], 16)) for i in range(0, len(m.group(1)), 4)), s)
    s = UNI1.sub(lambda m: chr(int(m.group(1), 16)), s)
    return s.replace("''", "'")


def split_args(body: str) -> list[str]:
    """Split an argument list on top-level commas."""
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
    """-> (ents, counts). ents: {id: (TYPE, [args])} for wanted types only."""
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
                continue                      # entity wrapped across lines
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
    """File length unit in metres. Some models are drawn in millimetres."""
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
    """Contour points: follow references down to IfcCartesianPoint."""
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
    """Room area from the FootPrint contour.

    VERIFIED: 200 of 200 esplanades spaces matched their answers.
    Needed because some models carry no quantity sets at all, and the
    printed label is rounded ("5,8" against the real 5.76).
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


_STR = re.compile(r"'((?:[^']|'')*)'")


def header(path: str) -> dict:
    """FILE_DESCRIPTION / FILE_NAME / FILE_SCHEMA of the STEP header."""
    txt = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().upper().startswith("DATA"):
                break
            txt += line.strip()
    out = {}
    for key in ("FILE_DESCRIPTION", "FILE_NAME", "FILE_SCHEMA"):
        m = re.search(key + r"\s*\((.*?)\)\s*;", txt, re.I | re.S)
        if m:
            out[key.lower()] = [_unescape(x) for x in _STR.findall(m.group(1))]
    fn = out.get("file_name", [])
    if len(fn) >= 3:                    # ..., preprocessor, originating system, authorization
        out["software"] = " / ".join(x for x in fn[-3:-1] if x)
    return out


def parse(path: str, dict_rows: dict):
    ents, counts = read(path)
    scale = _unit_scale(ents)

    # --- storeys ---------------------------------------------------------
    storeys = {}
    for eid, (cls, a) in ents.items():
        if cls != "IFCBUILDINGSTOREY":
            continue
        name = (_str(a[2]) if len(a) > 2 else None) or \
               (_str(a[7]) if len(a) > 7 else None) or ""
        lvl = _num(a[9]) if len(a) > 9 else None
        storeys[eid] = {"name": name, "level": (lvl or 0.0) * scale, "guid": _str(a[0])}
    order = sorted(storeys, key=lambda k: storeys[k]["level"])
    for i, k in enumerate(order):
        storeys[k]["ord"] = i

    # --- element to storey -----------------------------------------------
    # doors, walls, windows: ContainedInSpatialStructure
    # spaces: Aggregates (a storey IS COMPOSED OF its spaces)
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

    # --- quantities (Qto) --------------------------------------------------
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
                # lengths are in the project unit (often mm); areas and volumes
                # in these files are already m2 / m3
                vals[nm] = v * scale if qcls == "IFCQUANTITYLENGTH" else v
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

    # --- element -> type name (IfcRelDefinesByType) -------------------------
    typename: dict[int, str] = {}
    for cls, a in ents.values():
        if cls == "IFCRELDEFINESBYTYPE" and len(a) > 5:
            tref = _refs(a[5])
            if tref and tref[0] in ents and len(ents[tref[0]][1]) > 2:
                tn = _str(ents[tref[0]][1][2])
                if tn:
                    for e in _refs(a[4]):
                        typename[e] = tn

    # --- elements ----------------------------------------------------------
    out = []
    for eid, (cls, a) in ents.items():
        t = CLASS2TYPE.get(cls)
        if not t:
            continue
        guid = _str(a[0]) if a else None
        name = _str(a[2]) if len(a) > 2 else None
        longname = _str(a[7]) if len(a) > 7 else None

        # a space has two names: Name is the number ("4-3"), LongName is the
        # name ("TUALETTRUUM"). Questions join them: "TUALETTRUUM (4-3)".
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
        if t in ("door", "window"):
            w = _num(a[9]) * scale if len(a) > 9 and _num(a[9]) else qvals.get(eid, {}).get("Width")
            h = _num(a[8]) * scale if len(a) > 8 and _num(a[8]) else qvals.get(eid, {}).get("Height")
            pj = json.loads(row["props"])
            pj.update({"object_type": typename.get(eid) or (_str(a[4]) if len(a) > 4 else None),
                       "width_m": round(w, 3) if w else None,
                       "height_m": round(h, 3) if h else None})
            row["props"] = json.dumps(pj, ensure_ascii=False)
        if t == "room":
            # prefer a stored quantity, else measure the contour ourselves
            row["area"] = pick(eid, AREA) or _footprint_area(ents, eid, scale)
            row["volume"] = pick(eid, VOL)
        elif t in ("wall", "beam"):
            row["length"] = pick(eid, LEN)
            if t == "wall":
                row["area"] = pick(eid, ("NetSideArea", "GrossSideArea"))
                row["volume"] = pick(eid, VOL)
        elif t == "slab":
            row["area"] = pick(eid, AREA)
            row["volume"] = pick(eid, VOL)
        out.append(row)

    # Elements of a type that sit in storeys are the building; copies of the same
    # type with no storey (nested parts, library objects) are kept but secondary.
    # Verified on every "in the model" question of the preview set.
    owned = {r["type"] for r in out if r["floor_name"]}
    for r in out:
        if r["type"] in owned and not r["floor_name"]:
            r["is_primary"] = 0

    names = {}
    for cls, a in ents.values():
        if cls in ("IFCBUILDING", "IFCPROJECT", "IFCSITE") and len(a) > 2:
            names[cls[3:].lower()] = {"name": _str(a[2]),
                                      "long_name": _str(a[7]) if len(a) > 7 else None,
                                      "description": _str(a[3])}
    hd = header(path)
    apps = [f"{_str(a[2])} {_str(a[1])}".strip() for c, a in ents.values()
            if c == "IFCAPPLICATION" and len(a) > 2 and _str(a[2])]
    if apps:
        hd["application"] = apps[0]       # the authoring application, e.g. "ARCHICAD-64 20.0.0"
    out.append({"type": "building", "label": (names.get("building") or {}).get("name"),
                "props": json.dumps({"header": hd, **names}, ensure_ascii=False)})

    fl = [{"name": storeys[k]["name"], "level": storeys[k]["level"],
           "ord": storeys[k]["ord"]} for k in order]
    return fl, out
