#!/usr/bin/env python3
"""Compact facts sheet for one IFC model, built from the index (no model call).

    python ifc_facts.py --db ../app.db --drawing NAME

Free-form questions ("what is the space breakdown of the basement?") cannot be
a template. The model gets this sheet instead of the whole file and only
phrases the answer: every number in it was computed here.
"""
import argparse
import collections
import json
import sqlite3
import sys


def _p(row):
    try:
        return json.loads(row["props"] or "{}")
    except ValueError:
        return {}


def facts(con: sqlite3.Connection, drawing: str) -> dict:
    con.row_factory = sqlite3.Row
    d = con.execute("SELECT id FROM drawings WHERE name=? OR id=?", (drawing, drawing)).fetchone()
    if not d:
        return {}
    did = d["id"]
    floors = con.execute("SELECT id, name, level FROM floors WHERE drawing_id=? ORDER BY level, ord", (did,)).fetchall()
    fname = {f["id"]: f["name"] for f in floors}
    els = con.execute("SELECT * FROM elements WHERE drawing_id=? AND is_primary=1", (did,)).fetchall()

    out: dict = {}
    b = next((e for e in els if e["type"] == "building"), None)
    if b:
        bp = _p(b)
        out["building"] = {k: (v or {}).get("name") for k, v in bp.items() if k in ("project", "site", "building")}
        out["building"]["description"] = (bp.get("building") or {}).get("description")
        h = bp.get("header") or {}
        out["software"] = {"application": h.get("application"), "exporter": h.get("software")}
        out["ifc_schema"] = (h.get("file_schema") or [None])[0]

    levels = []
    for i, f in enumerate(floors):
        nxt = floors[i + 1]["level"] if i + 1 < len(floors) else None
        lv = {"name": f["name"], "elevation_m": f["level"]}
        if nxt is not None and f["level"] is not None:
            lv["height_to_next_m"] = round(nxt - f["level"], 3)
        levels.append(lv)
    out["levels"] = levels

    by_type = collections.Counter(e["type"] for e in els if e["type"] != "building")
    out["counts"] = dict(by_type)

    spaces = [e for e in els if e["type"] == "room"]
    per_floor = collections.defaultdict(list)
    for s in spaces:
        per_floor[fname.get(s["floor_id"], "?")].append(s)
    out["spaces_total"] = {"count": len(spaces), "area_m2": round(sum(s["area"] or 0 for s in spaces), 1)}
    out["spaces_by_level"] = {
        n: {
            "count": len(v),
            "area_m2": round(sum(s["area"] or 0 for s in v), 1),
            "spaces": [{"number": _p(s).get("number") or s["source_ref"], "name": s["label"],
                        "area_m2": round(s["area"], 1) if s["area"] is not None else None}
                       for v in [sorted(v, key=lambda s: str(_p(s).get("number") or s["label"]))] for s in v],
        }
        for n, v in per_floor.items()
    }
    names = collections.Counter((s["label"] or "").strip() for s in spaces)
    out["space_names"] = dict(names.most_common(25))
    big = sorted((s for s in spaces if s["area"]), key=lambda s: -s["area"])[:5]
    out["largest_spaces"] = [{"name": s["label"], "level": fname.get(s["floor_id"]), "area_m2": round(s["area"], 1)} for s in big]

    slabs = collections.defaultdict(float)
    for e in els:
        if e["type"] == "slab" and e["area"]:
            slabs[fname.get(e["floor_id"], "?")] += e["area"]
    out["slab_area_by_level_m2"] = {k: round(v, 1) for k, v in slabs.items()}

    for t in ("door", "window"):
        rows = [e for e in els if e["type"] == t]
        kinds = collections.Counter(_p(e).get("object_type") or e["subtype"] or "?" for e in rows)
        ws = [(_p(e).get("width_m"), e) for e in rows if _p(e).get("width_m")]
        out[t + "s"] = {"count": len(rows), "types": dict(kinds.most_common(15))}
        if ws:
            out[t + "s"]["width_m"] = {"min": min(w for w, _ in ws), "max": max(w for w, _ in ws)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--drawing", required=True)
    a = ap.parse_args()
    json.dump(facts(sqlite3.connect(a.db), a.drawing), sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    sys.exit(main())
