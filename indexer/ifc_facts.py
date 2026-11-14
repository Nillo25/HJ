#!/usr/bin/env python3
"""Compact facts sheet for one IFC model, built from the index (no model call).

    python ifc_facts.py --db index.db --drawing NAME [--q "question"]

Free-form questions ("what is the space breakdown of the basement?") cannot be a template. The model gets
this sheet instead of the whole file and only phrases the answer: every number in it was computed here.
The sheet is cut to the blocks the question is about, because input tokens are what inference costs.
"""
import argparse
import collections
import json
import re
import sqlite3
import sys

# question words -> blocks. "core" is always present; a question that matches nothing gets everything.
ROUTES = [
    ("software", r"software|standard|schema|version|created|generated|author|exporter|tool"),
    ("building", r"type of building|building'?s? type|function|situated|located|location|address|site|project|where"),
    ("levels", r"level|storey|story|floor|height|elevation|basement|ground|attic|roof level"),
    ("spaces", r"space|room|area|footprint|usable|program|ceiling|tallest|largest|biggest|smallest|percent"),
    ("doors", r"door"),
    ("windows", r"window"),
    ("walls", r"wall|material|thick|load.?bearing|partition|exterior|interior|outer|inner|outside|inside"),
    ("slabs", r"slab|floor slab|volume|plate"),
    ("coverings", r"covering|ceiling|finish|cladding|flooring"),
    ("furniture", r"furniture|furnishing|equipment item|inventory|fixture|sanitary|toilet|sink|bath|wc"),
    ("stairs", r"stair|ramp|railing|balustrade|handrail"),
    ("structure", r"structur|beam|column|girder|truss|foundation|pillar|roof"),
    ("mep", r"mep|pipe|duct|heating|ventilat|plumb|electric|hvac|system|installation|parking|equipment|lighting|sprinkler"),
    ("fire", r"fire|rating|fire.?resist"),
]
TOKENS_CAP = 6000          # characters of JSON per block list are cut to keep the sheet small


def _p(row):
    try:
        return json.loads(row["props"] or "{}")
    except ValueError:
        return {}


def _norm_name(s):
    """'Seminarraum (006)' -> 'Seminarraum': a function name without the room number."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", (s or "").strip()) or "?"


def _flag(psets, rx):
    """First boolean-ish property whose name matches rx -> True / False / None."""
    for k, v in (psets or {}).items():
        if re.search(rx, k, re.I):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                lv = v.lower()
                if lv in ("true", "ja", "yes", "wahr", "tragend"):
                    return True
                if lv in ("false", "nein", "no", "falsch", "nicht tragend"):
                    return False
    return None


def _external(e, p):
    f = _flag(p.get("psets"), r"isexternal|exterior|extern(?!e ifc)")
    if f is not None:
        return f
    n = (p.get("type_name") or p.get("name") or "").lower()
    if re.search(r"auss|außen|exterior|outer|facade|fassade|\bext\b|\bav\b", n):
        return True
    if re.search(r"innen|interior|inner|partition|\bint\b|\biv\b|trenn", n):
        return False
    return None


def _by_type(rows, withdims=True, limit=14):
    """Aggregate elements by type name: count, material, thickness, flags, totals."""
    g = collections.OrderedDict()
    for e in rows:
        p = _p(e)
        k = (p.get("type_name") or p.get("name") or e["subtype"] or "?")
        d = g.setdefault(k, {"count": 0})
        d["count"] += 1
        if p.get("materials") and "materials" not in d:
            d["materials"] = list(dict.fromkeys(p["materials"]))[:4]
        if p.get("thickness_mm") and "thickness_mm" not in d:
            d["thickness_mm"] = p["thickness_mm"]
        ext = _external(e, p)
        if ext is not None:
            d.setdefault("external", ext)
        lb = _flag(p.get("psets"), r"load.?bearing|tragend")
        if lb is not None:
            d["load_bearing_count"] = d.get("load_bearing_count", 0) + (1 if lb else 0)
        fr = {k2.split(".")[-1]: v for k2, v in (p.get("psets") or {}).items()
              if re.search(r"fire|feuer|brand", k2, re.I) and v not in ("", None, False)}
        if fr and "fire" not in d:
            d["fire"] = fr
        if withdims:
            for fld, key in (("length", "length_m"), ("area", "area_m2"), ("volume", "volume_m3")):
                if e[fld]:
                    d[key] = round(d.get(key, 0) + e[fld], 2)
            for fld in ("width_m", "height_m"):
                if p.get(fld):
                    lo, hi = d.get(fld + "_range", [p[fld], p[fld]])
                    d[fld + "_range"] = [min(lo, p[fld]), max(hi, p[fld])]
    items = sorted(g.items(), key=lambda kv: -kv[1]["count"])
    out = {k: v for k, v in items[:limit]}
    if len(items) > limit:
        out["(other types)"] = {"count": sum(v["count"] for _k, v in items[limit:]), "kinds": len(items) - limit}
    return out


def _length(e):
    """Wall length: the stored quantity, else the longest extent of its footprint (a wall is a long thin box)."""
    if e["length"]:
        return e["length"], False
    try:
        pts = json.loads(e["polygon"] or "[]")
    except ValueError:
        pts = []
    if len(pts) < 2:
        return 0.0, False
    best = max(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for i, a in enumerate(pts) for b in pts[i + 1:])
    return best, True


def select(question):
    q = (question or "").lower()
    blocks = [b for b, rx in ROUTES if re.search(rx, q)]
    return blocks or [b for b, _ in ROUTES]


def facts(con: sqlite3.Connection, drawing: str, question: str = None) -> dict:
    con.row_factory = sqlite3.Row
    d = con.execute("SELECT id FROM drawings WHERE name=? OR id=?", (drawing, drawing)).fetchone()
    if not d:
        return {}
    did = d["id"]
    floors = con.execute("SELECT id, name, level FROM floors WHERE drawing_id=? ORDER BY level, ord", (did,)).fetchall()
    fname = {f["id"]: f["name"] for f in floors}
    els = con.execute("SELECT * FROM elements WHERE drawing_id=? AND is_primary=1", (did,)).fetchall()
    want = set(select(question))
    full_list = bool(re.search(r"list|name|each|all|complete|detail", (question or "").lower()))

    out: dict = {}
    b = next((e for e in els if e["type"] == "building"), None)
    bp = _p(b) if b else {}
    h = bp.get("header") or {}
    # core: small, always sent
    out["model"] = {k: (v or {}).get("name") for k, v in bp.items() if k in ("project", "site", "building")}
    out["levels"] = []
    for i, f in enumerate(floors):
        nxt = floors[i + 1]["level"] if i + 1 < len(floors) else None
        lv = {"name": f["name"], "elevation_m": f["level"]}
        if nxt is not None and f["level"] is not None:
            lv["height_to_next_m"] = round(nxt - f["level"], 3)
        out["levels"].append(lv)
    out["counts"] = dict(collections.Counter(e["type"] for e in els if e["type"] != "building"))

    if want & {"software", "building"}:
        out["software"] = {"application": h.get("application"), "exporter": h.get("software")}
        out["ifc_schema"] = (h.get("file_schema") or [None])[0]
        out["building"] = {"description": (bp.get("building") or {}).get("description"),
                           "long_name": (bp.get("building") or {}).get("long_name"),
                           "site_long_name": (bp.get("site") or {}).get("long_name"),
                           "project_description": (bp.get("project") or {}).get("description"),
                           "site_latitude": (bp.get("site") or {}).get("latitude"),
                           "site_longitude": (bp.get("site") or {}).get("longitude")}

    spaces = [e for e in els if e["type"] == "room"]
    if want & {"spaces", "levels"}:
        per_floor = collections.defaultdict(list)
        for s in spaces:
            per_floor[fname.get(s["floor_id"], "?")].append(s)
        total_area = sum(s["area"] or 0 for s in spaces)
        out["spaces_total"] = {"count": len(spaces), "area_m2": round(total_area, 1)}
        slabs = collections.defaultdict(float)
        for e in els:
            if e["type"] == "slab" and e["area"]:
                slabs[fname.get(e["floor_id"], "?")] += e["area"]
        if slabs:
            out["slab_area_by_level_m2"] = {k: round(v, 1) for k, v in slabs.items()}
        lvls = {}
        for n, v in per_floor.items():
            area = sum(s["area"] or 0 for s in v)
            fn = collections.defaultdict(lambda: [0, 0.0])
            for s in v:
                x = fn[_norm_name(s["label"])]
                x[0] += 1
                x[1] += s["area"] or 0
            lvls[n] = {
                "count": len(v), "area_m2": round(area, 1),
                "by_function": {k: {"count": c, "area_m2": round(a, 1), "percent": round(100 * a / area, 1) if area else None}
                                for k, (c, a) in sorted(fn.items(), key=lambda kv: -kv[1][1])[:14]},
            }
            if full_list or len(v) <= 8:
                lvls[n]["spaces"] = [{"number": _p(s).get("number") or _p(s).get("name"), "name": _norm_name(s["label"]),
                                      "area_m2": round(s["area"], 1) if s["area"] is not None else None}
                                     for s in sorted(v, key=lambda s: str(_p(s).get("number") or s["label"]))]
        out["spaces_by_level"] = lvls
        big = sorted((s for s in spaces if s["area"]), key=lambda s: -s["area"])[:5]
        out["largest_spaces"] = [{"number": _p(s).get("number"), "name": _norm_name(s["label"]), "level": fname.get(s["floor_id"]),
                                  "area_m2": round(s["area"], 1), "height_m": _p(s).get("height_m")} for s in big]
        tall = sorted((s for s in spaces if _p(s).get("height_m")), key=lambda s: -_p(s)["height_m"])[:10]
        if tall:
            out["tallest_spaces"] = [{"name": _norm_name(s["label"]), "number": _p(s).get("number"), "level": fname.get(s["floor_id"]),
                                      "height_m": _p(s)["height_m"]} for s in tall]
        fp = [s["area"] for s in spaces if s["area"] and floors and fname.get(s["floor_id"]) == floors[0]["name"]]
        out["ground_floor_spaces_area_m2"] = round(sum(fp), 1) if fp else None

    def rows(*types):
        return [e for e in els if e["type"] in types]

    if "doors" in want:
        r = rows("door")
        out["doors"] = {"count": len(r), "by_type": _by_type(r)}
        ws = sorted(((_p(e).get("width_m"), e) for e in r if _p(e).get("width_m")), key=lambda x: x[0])
        if ws:
            out["doors"]["narrowest"] = [{"name": _p(e).get("name"), "width_m": w, "type": _p(e).get("type_name")} for w, e in ws[:2]]
            out["doors"]["widest"] = [{"name": _p(e).get("name"), "width_m": w, "type": _p(e).get("type_name")} for w, e in ws[-3:][::-1]]
    if "windows" in want:
        r = rows("window")
        out["windows"] = {"count": len(r), "by_type": _by_type(r)}
        ws = sorted(((_p(e).get("width_m"), e) for e in r if _p(e).get("width_m")), key=lambda x: x[0])
        if ws:
            out["windows"]["narrowest"] = [{"name": _p(e).get("name"), "width_m": w} for w, e in ws[:2]]
            out["windows"]["widest"] = [{"name": _p(e).get("name"), "width_m": w} for w, e in ws[-2:][::-1]]
    if "walls" in want:
        r = rows("wall")
        ext = [e for e in r if _external(e, _p(e)) is True]
        inn = [e for e in r if _external(e, _p(e)) is False]
        lb = [_flag(_p(e).get("psets"), r"load.?bearing|tragend") for e in r]
        lens = {e["id"]: _length(e) for e in r}
        est = any(v[1] for v in lens.values())
        L = lambda xs: round(sum(lens[e["id"]][0] for e in xs), 1)
        out["walls"] = {
            "count": len(r), "by_type": _by_type(r),
            "total_length_m": L(r),
            "exterior": {"count": len(ext), "length_m": L(ext)},
            "interior": {"count": len(inn), "length_m": L(inn)},
            "length_note": "measured from the wall footprints (longest extent), not a stored quantity" if est else None,
            "load_bearing": {"yes": lb.count(True), "no": lb.count(False), "unknown": lb.count(None),
                             "percent_yes": round(100 * lb.count(True) / len(r), 1) if r else None},
        }
    if "slabs" in want:
        r = rows("slab")
        roof = [e for e in r if str(_p(e).get("predefined", "")).upper() == "ROOF"]
        flo = [e for e in r if e not in roof]
        vol = lambda xs: round(sum(e["volume"] or 0 for e in xs), 2)
        out["slabs"] = {"count": len(r), "total_volume_m3": vol(r),
                        "floor_slabs": {"count": len(flo), "volume_m3": vol(flo),
                                        "volume_range_m3": [round(min((e["volume"] for e in flo if e["volume"]), default=0), 2),
                                                            round(max((e["volume"] for e in flo if e["volume"]), default=0), 2)]},
                        "roof_slabs": {"count": len(roof), "volume_m3": vol(roof)}, "by_type": _by_type(r)}
    if "coverings" in want:
        out["coverings"] = {"count": len(rows("covering")), "by_type": _by_type(rows("covering"))}
    if "furniture" in want:
        out["furniture"] = {"count": len(rows("furniture")), "by_type": _by_type(rows("furniture"), False, 18)}
        fx = rows("fixture")
        out["sanitary_fixtures"] = {"count": len(fx), "by_type": _by_type(fx, False)}
        g = [e for e in fx + rows("furniture") if re.search(r"wc|toilet|sink|basin|wasch|dusch|bad|lavabo", (_p(e).get("type_name") or "") + (e["label"] or ""), re.I)]
        out["sanitary_by_level"] = dict(collections.Counter(fname.get(e["floor_id"], "?") for e in g))
        wcs = [e for e in spaces if e["subtype"] in ("wc",) or re.search(r"\bwc\b|toilet|toilette", e["label"] or "", re.I)]
        out["wc_spaces_by_level"] = dict(collections.Counter(fname.get(e["floor_id"], "?") for e in wcs))
    if "stairs" in want:
        out["stairs"] = {"count": len(rows("stair")), "by_type": _by_type(rows("stair"), False)}
        out["ramps"] = {"count": len(rows("ramp"))}
        out["railings"] = {"count": len(rows("railing")), "by_type": _by_type(rows("railing"), False)}
    if "structure" in want:
        out["structure"] = {t: {"count": len(rows(t)), "by_type": _by_type(rows(t), False, 10)} for t in ("column", "beam", "slab", "roof") if rows(t)}
    if "mep" in want:
        sysnames = collections.Counter()
        for e in els:
            for sname in _p(e).get("systems") or []:
                sysnames[sname] += 1
        out["mep"] = {t: {"count": len(rows(t)), "by_type": _by_type(rows(t), False, 10)} for t in ("pipe", "duct", "equipment", "fixture") if rows(t)}
        if sysnames:
            out["mep_systems"] = dict(sysnames.most_common(12))
        out["proxy_elements"] = {"count": len(rows("proxy")), "by_type": _by_type(rows("proxy"), False, 8)} if rows("proxy") else None
    if "fire" in want:
        fr = collections.defaultdict(lambda: collections.Counter())
        for e in els:
            for k, v in (_p(e).get("psets") or {}).items():
                if re.search(r"fire|feuer|brand", k, re.I) and v not in ("", None, False):
                    fr[e["type"]][f"{_p(e).get('type_name') or '?'}: {v}"] += 1
        out["fire_ratings"] = {t: dict(c.most_common(8)) for t, c in fr.items()}
        out["fire_rating_note"] = "empty when the model carries no fire-rating property"
    return {k: v for k, v in out.items() if v not in (None, {}, [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--drawing", required=True)
    ap.add_argument("--q", default=None, help="question: the sheet is cut to the blocks it is about")
    a = ap.parse_args()
    json.dump(facts(sqlite3.connect(a.db), a.drawing, a.q), sys.stdout, ensure_ascii=False, default=str, separators=(",", ":"))


if __name__ == "__main__":
    sys.exit(main())
