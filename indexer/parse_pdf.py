"""Vector PDF -> schedule rows (door schedule, finish room schedule).

Reads text lines with coordinates through pdfminer.six (pure Python), groups
them into visual rows by baseline, then assigns each token to a column by its
x position. Column bounds are tuned for the A-801 layout in the preview set;
a row that does not match the expected shape is skipped, never guessed.
"""
import json
import re

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextLine

DOOR_NO = re.compile(r"^\d{3}\.[A-Z]\d?$")
AREA_SF = re.compile(r"^([\d,]+)\s*SF$")

# door schedule columns: (field, x_from, x_to)
DOOR_COLS = [
    ("number", 0, 195), ("room_name", 195, 310), ("room", 310, 360),
    ("width", 360, 425), ("height", 425, 475), ("type", 475, 520),
    ("door_matl", 520, 590), ("frame_matl", 590, 630), ("frame_type", 630, 690),
    ("hardware", 690, 780), ("remark", 780, 830), ("notes", 830, 900),
]
# finish room schedule columns
ROOM_COLS = [
    ("name", 245, 425), ("floor", 425, 590), ("base", 590, 640),
    ("north", 640, 720), ("south", 720, 805), ("east", 805, 890),
    ("west", 890, 970), ("ceiling", 970, 1100),
]
DOOR_X_MAX = 950      # the legend to the right shares the same baselines


MAX_PAGES = 30          # a drawing set can have hundreds of pages: the first ones are read, the rest is said not to be
MAX_SPANS_PER_PAGE = 6000


def _lines(path):
    for pno, pg in enumerate(extract_pages(path, maxpages=MAX_PAGES), 1):
        out = []

        def walk(o):
            if isinstance(o, LTTextLine):
                t = o.get_text().strip()
                if t:
                    out.append((o.x0, o.y0, o.x1, o.y1, t))
            elif hasattr(o, "__iter__"):
                for c in o:
                    walk(c)
        walk(pg)
        yield pno, out


def _rows(lines, tol=3.0):
    """Group tokens whose baselines are within tol of each other."""
    rows = []
    for ln in sorted(lines, key=lambda l: -l[1]):
        for r in rows:
            if abs(r[0] - ln[1]) <= tol:
                r[1].append(ln)
                break
        else:
            rows.append([ln[1], [ln]])
    return [sorted(r[1], key=lambda l: l[0]) for r in rows]


def _col(x, cols):
    for name, a, b in cols:
        if a <= x < b:
            return name
    return None


def _feet(s):
    """3' - 0" -> 36.0 inches; 2' - 11 3/16" -> 35.1875. None if unparseable."""
    m = re.match(r"^(\d+)'\s*-\s*(\d+)(?:\s+(\d+)/(\d+))?\"$", s.strip())
    if not m:
        return None
    ft, inch, n, d = m.groups()
    return int(ft) * 12 + int(inch) + (int(n) / int(d) if n else 0)


def parse(path: str, dict_rows: dict):
    out = []
    for pno, lines in _lines(path):
        # every line of text with its position: this is what free-form questions are answered from
        for x0, y0, x1, y1, t in lines[:MAX_SPANS_PER_PAGE]:
            out.append({"type": "text_span", "label": t, "page": pno,
                        "bbox": json.dumps([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)])})
        for row in _rows(lines):
            # margin grid markers ("E", "D") sit left of the table on some baselines
            row = [l for l in row if l[0] > 100] or row
            head = row[0]
            txt = head[4]

            if DOOR_NO.match(txt) and head[0] < 200:
                f = {}
                for x0, y0, x1, y1, t in row:
                    if x0 >= DOOR_X_MAX:
                        continue
                    k = _col(x0, DOOR_COLS)
                    if k:
                        f[k] = f[k] + " " + t if k in f else t
                if not f.get("type") and not f.get("width"):
                    continue
                out.append({
                    "type": "door_row", "label": f["number"],
                    "label_norm": f["number"].lower(), "page": pno,
                    "bbox": json.dumps([head[0], head[1],
                                        max(l[2] for l in row if l[0] < DOOR_X_MAX),
                                        head[3]]),
                    "source_ref": f"door:{f['number']}",
                    "props": json.dumps({**f, "width_in": _feet(f.get("width", "")),
                                         "height_in": _feet(f.get("height", ""))},
                                        ensure_ascii=False),
                })
                continue

            m = AREA_SF.match(txt)
            redundant = txt == "Redundant"      # listed room, area not given
            if (m or redundant) and head[0] < 200 and len(row) > 2 and row[1][4].isdigit():
                f = {}
                for x0, y0, x1, y1, t in row:
                    k = _col(x0, ROOM_COLS)
                    if k:
                        f[k] = f[k] + " " + t if k in f else t
                out.append({
                    "type": "room_row", "label": row[1][4],
                    "label_norm": row[1][4], "page": pno,
                    "area": float(m.group(1).replace(",", "")) if m else None,
                    "bbox": json.dumps([head[0], head[1], row[-1][2], head[3]]),
                    "source_ref": f"room:{row[1][4]}",
                    "props": json.dumps({**f, "unit": "SF"}, ensure_ascii=False),
                })
    return out
