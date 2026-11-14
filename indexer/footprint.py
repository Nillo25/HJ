"""Gross footprint of a floor: the area inside its outer walls, walls included.

No geometry library: the room, wall, door and window polygons are drawn into a bitmap (one Python int per row),
tiny gaps are closed (a door opening, a hairline between two polygons), the outside is flooded from the border,
and whatever the flood cannot reach is the building. Courtyards that open to the outside are outside.
"""
import math


def _fill_rows(poly, rows, x0, y0, cell, w, h):
    n = len(poly)
    if n < 3:
        return
    ys = [p[1] for p in poly]
    r0 = max(int((min(ys) - y0) / cell), 0)
    r1 = min(int((max(ys) - y0) / cell) + 1, h - 1)
    for r in range(r0, r1 + 1):
        y = y0 + (r + 0.5) * cell
        xs = []
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            if (a[1] > y) != (b[1] > y):
                xs.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        xs.sort()
        for j in range(0, len(xs) - 1, 2):
            c0 = max(int(math.floor((xs[j] - x0) / cell)), 0)
            c1 = min(int(math.ceil((xs[j + 1] - x0) / cell)), w)
            if c1 > c0:
                rows[r] |= ((1 << (c1 - c0)) - 1) << c0


def _dilate(rows, w):
    full = (1 << w) - 1
    out = []
    for y, r in enumerate(rows):
        v = r | (r << 1) | (r >> 1)
        if y > 0:
            v |= rows[y - 1]
        if y + 1 < len(rows):
            v |= rows[y + 1]
        out.append(v & full)
    return out


def _erode(rows, w):
    full = (1 << w) - 1
    out = []
    for y, r in enumerate(rows):
        v = r & (r << 1) & (r >> 1)
        v &= rows[y - 1] if y > 0 else 0
        v &= rows[y + 1] if y + 1 < len(rows) else 0
        out.append(v & full)
    return out


def gross_area(polys, cell=0.15, close=2):
    """Area in the unit of the polygons squared. Returns (area, bbox of the footprint) or (None, None)."""
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return None, None
    pad = (close + 2) * cell
    xs = [x for p in polys for x, _ in p]
    ys = [y for p in polys for _, y in p]
    x0, y0 = min(xs) - pad, min(ys) - pad
    w = int(math.ceil((max(xs) + pad - x0) / cell)) + 1
    h = int(math.ceil((max(ys) + pad - y0) / cell)) + 1
    if w * h > 4_000_000:                       # a plan in millimetres would need a coarser grid: do not guess
        return None, None
    rows = [0] * h
    for p in polys:
        _fill_rows(p, rows, x0, y0, cell, w, h)
    for _ in range(close):
        rows = _dilate(rows, w)
    for _ in range(close):
        rows = _erode(rows, w)

    full = (1 << w) - 1
    free = [~r & full for r in rows]
    ext = [0] * h
    ext[0], ext[-1] = free[0], free[-1]
    for y in range(h):
        ext[y] |= free[y] & 1 | free[y] & (1 << (w - 1))
    changed = True
    while changed:                               # spread one cell per round until the outside stops growing
        changed = False
        for y in list(range(h)) + list(range(h - 1, -1, -1)):
            e = ext[y]
            v = e | (e << 1) | (e >> 1)
            if y > 0:
                v |= ext[y - 1]
            if y + 1 < h:
                v |= ext[y + 1]
            v &= free[y]
            if v != e:
                ext[y] = v
                changed = True
    inside = sum(bin(full & ~ext[y]).count("1") for y in range(h))
    ones = [(y, full & ~ext[y]) for y in range(h) if full & ~ext[y]]
    if not ones:
        return None, None
    cols = 0
    for _, r in ones:
        cols |= r
    low = (cols & -cols).bit_length() - 1
    high = cols.bit_length() - 1
    bb = [x0 + low * cell, y0 + ones[0][0] * cell, x0 + (high + 1) * cell, y0 + (ones[-1][0] + 1) * cell]
    return inside * cell * cell, bb
