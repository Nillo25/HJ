#!/usr/bin/env python3
"""Index the Stoa dataset into SQLite.

    python index.py <dataset folder> [--db app.db] [--only swiss_dwellings]

Stores EVERYTHING found in a file, regardless of the questions.
Sources with no questions (europeana, finna, habs, wikimedia) and formats
with neither questions nor use as a source (svg, dwg, jpg) are skipped.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import time

import labels as L

IGNORE_SOURCES = {"europeana", "finna", "habs", "wikimedia"}
IGNORE_EXT = {".svg", ".dwg", ".jpg", ".jpeg", ".md", ".ds_store"}

# what answers what: format -> role
ROLE = {".dxf": "truth", ".ifc": "truth", ".pdf": "truth", ".png": "display"}


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def capabilities(formats: set) -> dict:
    if ".dxf" in formats or ".ifc" in formats:
        return {"counts": "exact", "areas": "exact", "text": "exact",
                "reason": "geometry available (DXF/IFC)"}
    if ".pdf" in formats:
        return {"counts": "exact", "areas": "none", "text": "exact",
                "reason": "vector PDF: text and tables, no room geometry"}
    if ".png" in formats:
        return {"counts": "estimated", "areas": "none", "text": "estimated",
                "reason": "raster only, read by vision"}
    return {"counts": "none", "areas": "none", "text": "none",
            "reason": "format not supported"}


def collect(root: str, only: str | None):
    """-> {(source, basename): [paths]}"""
    groups: dict[tuple[str, str], list[str]] = {}
    data = os.path.join(root, "data")
    base = data if os.path.isdir(data) else root
    for src in sorted(os.listdir(base)):
        d = os.path.join(base, src)
        if not os.path.isdir(d) or src in IGNORE_SOURCES:
            continue
        if only and src != only:
            continue
        for dirpath, _, names in os.walk(d):
            for n in names:
                ext = os.path.splitext(n)[1].lower()
                if ext in IGNORE_EXT:
                    continue
                stem = os.path.splitext(n)[0]
                groups.setdefault((src, stem), []).append(os.path.join(dirpath, n))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--db", default="app.db")
    ap.add_argument("--only", default=None, help="single source, e.g. swiss_dwellings")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    con = sqlite3.connect(a.db)
    con.executescript(open(os.path.join(os.path.dirname(__file__), "schema.sql")).read())
    L.seed_dict(con)
    con.commit()
    dict_rows = L.load_dict(con)

    groups = collect(a.root, a.only)
    if a.limit:
        groups = dict(list(groups.items())[: a.limit])
    print(f"drawings to parse: {len(groups)}")

    t0 = time.time()
    done = failed = 0

    for (src, stem), paths in groups.items():
        con.execute(
            "INSERT OR IGNORE INTO drawings(name, source, status) VALUES (?,?,'indexing')",
            (stem, src))
        did = con.execute("SELECT id FROM drawings WHERE name=?", (stem,)).fetchone()[0]

        formats = set()
        exts = {os.path.splitext(p)[1].lower() for p in paths}
        for p in sorted(paths):
            ext = os.path.splitext(p)[1].lower()
            formats.add(ext)
            if ext == ".pdf" and exts & {".dxf", ".ifc"}:
                continue          # the PDF next to a DXF or IFC is the same drawing rendered: reading it too would count every label twice
            try:
                digest = sha256(p)
            except OSError as e:
                print(f"  ! {p}: {e}")
                continue
            if con.execute("SELECT 1 FROM files WHERE sha256=?", (digest,)).fetchone():
                continue          # same file already parsed
            fid = con.execute(
                "INSERT INTO files(drawing_id, path, sha256, format, role) VALUES (?,?,?,?,?)",
                (did, p, digest, ext.lstrip("."), ROLE.get(ext, "ignored"))).lastrowid

            rows, floors = [], []
            try:
                if ext == ".dxf":
                    import parse_dxf
                    rows = parse_dxf.parse(p, dict_rows)
                elif ext == ".ifc":
                    import parse_ifc
                    floors, rows = parse_ifc.parse(p, dict_rows)
                elif ext == ".pdf":
                    try:
                        import parse_pdf
                        rows = parse_pdf.parse(p, dict_rows)
                    except ModuleNotFoundError:
                        con.execute("UPDATE files SET role='display' WHERE id=?", (fid,))
                        continue
                else:
                    continue      # png: separate pass, see index_raster.py
            except Exception as e:
                con.execute("UPDATE files SET status='failed', note=? WHERE id=?",
                            (f"{type(e).__name__}: {e}", fid))
                failed += 1
                print(f"  ! {os.path.basename(p)}: {type(e).__name__}: {e}")
                continue

            floor_ids = {}
            for fl in floors:
                fl_id = con.execute(
                    "INSERT INTO floors(drawing_id, file_id, name, level, ord) VALUES (?,?,?,?,?)",
                    (did, fid, fl["name"], fl["level"], fl["ord"])).lastrowid
                floor_ids[fl["name"]] = fl_id

            for r in rows:
                if r["type"] == "pdf_info":              # what the PDF is made of: kept on the file, not as an object
                    con.execute("UPDATE files SET note=? WHERE id=?", (r["props"], fid))
                    continue
                if r["type"] == "text_span":
                    con.execute(
                        "INSERT INTO text_spans(file_id, page, text, bbox) VALUES (?,?,?,?)",
                        (fid, r.get("page"), r["label"], r["bbox"]))
                    continue
                con.execute(
                    """INSERT INTO elements
                       (drawing_id, file_id, floor_id, type, subtype, label, label_norm,
                        area, length, volume, bbox, polygon, page, source_ref,
                        confidence, is_primary, props)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (did, fid, floor_ids.get(r.get("floor_name")),
                     r["type"], r.get("subtype"), r.get("label"), r.get("label_norm"),
                     r.get("area"), r.get("length"), r.get("volume"),
                     r.get("bbox"), r.get("polygon"), r.get("page"), r.get("source_ref"),
                     r.get("confidence", 1.0), r.get("is_primary", 1), r.get("props")))

        con.execute("UPDATE drawings SET status='ready', capabilities=? WHERE id=?",
                    (json.dumps(capabilities(formats), ensure_ascii=False), did))
        done += 1
        if done % 20 == 0:
            con.commit()
            print(f"  {done}/{len(groups)}  {time.time()-t0:.0f}s")

    con.commit()

    n_el = con.execute("SELECT COUNT(*) FROM elements").fetchone()[0]
    print(f"\ndone: drawings {done}, failed {failed}, elements {n_el}, "
          f"{time.time()-t0:.0f}s")
    for row in con.execute(
            "SELECT type, COUNT(*) FROM elements GROUP BY type ORDER BY 2 DESC"):
        print(f"  {row[0]:12s} {row[1]}")


if __name__ == "__main__":
    main()
