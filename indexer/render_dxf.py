#!/usr/bin/env python3
"""DXF -> standalone SVG that looks like the drawing: walls, door swings, windows, fixtures, room labels.

    python render_dxf.py <file.dxf> <out.svg>

Coordinates are the DXF ones (metres) with Y negated, so a stored object at (x, y) is drawn at (x, -y) and
the viewer can lay its own shapes over this picture without any transform. Only the entities the swiss
plans use are drawn: LWPOLYLINE, LINE, ARC, CIRCLE, TEXT, and polygon HATCH boundaries.
"""
import math
import sys
from xml.sax.saxutils import escape

import parse_dxf as P

# layer -> (fill, stroke, stroke width in metres, dash)
STYLE = {
    "ROOM_OUTLINE": ("#f6f8fb", "#cfd6e0", 0.02, None),
    "UNIT_OUTLINE": ("none", "#98a2b3", 0.05, "0.3 0.15"),
    "WALL": ("#2f3744", "#2f3744", 0.02, None),
    "WINDOW": ("#d6ecfb", "#1f7fbf", 0.03, None),
    "DOOR": ("none", "#8a5a1f", 0.03, None),
    "FIXTURE": ("none", "#6d5bb0", 0.025, None),
    "ANNOTATION": ("none", "#667085", 0.02, None),
}
DEFAULT = ("none", "#667085", 0.02, None)
ORDER = ["ROOM_OUTLINE", "UNIT_OUTLINE", "WALL", "WINDOW", "DOOR", "FIXTURE", "ANNOTATION"]


def _f(e, code, default=0.0):
    try:
        return float((e.get(code) or [default])[0])
    except (TypeError, ValueError):
        return default


def _path(points, closed):
    d = "M" + "L".join(f"{x:.3f} {-y:.3f}" for x, y in points)
    return d + ("Z" if closed else "")


def _hatch_loops(e):
    """Polygon boundary paths of a HATCH (vertex count 93 per loop); edge-type boundaries are skipped."""
    counts = [int(float(v)) for v in e.get("93", [])]
    # the first 10/20 pair of a HATCH is its elevation point (0, 0), not a vertex
    xs = [float(v) for v in e.get("10", [])][1:]
    ys = [float(v) for v in e.get("20", [])][1:]
    out, i = [], 0
    for n in counts:
        if n < 3 or i + n > min(len(xs), len(ys)):
            break
        out.append(list(zip(xs[i:i + n], ys[i:i + n])))
        i += n
    return out


def _arc(e):
    cx, cy, r = _f(e, "10"), _f(e, "20"), _f(e, "40")
    a0, a1 = _f(e, "50"), _f(e, "51")
    if r <= 0:
        return None
    sweep = (a1 - a0) % 360 or 360
    p0 = (cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
    p1 = (cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))
    # y is negated, which turns a counter-clockwise arc into a counter-clockwise arc on screen: sweep flag 0
    return f"M{p0[0]:.3f} {-p0[1]:.3f}A{r:.3f} {r:.3f} 0 {1 if sweep > 180 else 0} 0 {p1[0]:.3f} {-p1[1]:.3f}"


def render(path):
    ents = P.read_entities(path)
    by_layer = {}
    texts = []
    xs, ys = [], []

    def grow(points):
        for x, y in points:
            xs.append(x)
            ys.append(-y)

    for e in ents:
        lay = P.layer(e)
        t = e["type"]
        if t == "LWPOLYLINE":
            p = P.pts(e)
            if len(p) >= 2:
                closed = bool(int(float((e.get("70") or [0])[0])) & 1)
                by_layer.setdefault(lay, []).append(("path", _path(p, closed or lay in ("WALL", "WINDOW", "ROOM_OUTLINE", "UNIT_OUTLINE"))))
                grow(p)
        elif t == "LINE":
            p = [(_f(e, "10"), _f(e, "20")), (_f(e, "11"), _f(e, "21"))]
            by_layer.setdefault(lay, []).append(("path", _path(p, False)))
            grow(p)
        elif t == "ARC":
            d = _arc(e)
            if d:
                by_layer.setdefault(lay, []).append(("arc", d))
                r = _f(e, "40")
                grow([(_f(e, "10") - r, _f(e, "20") - r), (_f(e, "10") + r, _f(e, "20") + r)])
        elif t == "CIRCLE":
            cx, cy, r = _f(e, "10"), _f(e, "20"), _f(e, "40")
            if r > 0:
                by_layer.setdefault(lay, []).append(("circle", f'cx="{cx:.3f}" cy="{-cy:.3f}" r="{r:.3f}"'))
                grow([(cx - r, cy - r), (cx + r, cy + r)])
        elif t == "HATCH":
            for loop in _hatch_loops(e):
                by_layer.setdefault(lay, []).append(("hatch", _path(loop, True)))
                grow(loop)
        elif t in ("TEXT", "MTEXT"):
            txt = (e.get("1") or [""])[0]
            if t == "MTEXT":
                txt = txt.replace("\\P", " ")
            if txt.strip():
                texts.append((_f(e, "10"), _f(e, "20"), max(_f(e, "40", 0.25), 0.1), txt.strip(), lay))

    if not xs:
        raise ValueError("nothing to draw")
    pad = 1.0
    x0, y0, x1, y1 = min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
    w, h = x1 - x0, y1 - y0

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.3f} {y0:.3f} {w:.3f} {h:.3f}">',
           f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{w:.3f}" height="{h:.3f}" fill="#ffffff"/>']
    for lay in ORDER + sorted(set(by_layer) - set(ORDER)):
        items = by_layer.get(lay)
        if not items:
            continue
        fill, stroke, sw, dash = STYLE.get(lay, DEFAULT)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<g stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"{da} fill="{fill}">')
        for kind, d in items:
            if kind == "path":
                out.append(f'<path d="{d}"/>')
            elif kind == "hatch":
                out.append(f'<path d="{d}" fill="#2f3744" stroke="none"/>')
            elif kind == "arc":
                out.append(f'<path d="{d}" fill="none" stroke-width="{sw * 0.6:.3f}" stroke-opacity="0.8"/>')
            else:
                out.append(f"<circle {d}/>")
        out.append("</g>")
    out.append('<g font-family="Arial, Helvetica, sans-serif" text-anchor="middle" fill="#1f2937">')
    for x, y, size, txt, lay in texts:
        col = "#667085" if lay == "ANNOTATION" else "#1f2937"
        out.append(f'<text x="{x:.3f}" y="{-y + size * 0.35:.3f}" font-size="{size:.3f}" fill="{col}">{escape(txt)}</text>')
    out.append("</g></svg>")
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: render_dxf.py file.dxf out.svg")
    svg = render(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"{len(svg) // 1024} KB")


if __name__ == "__main__":
    main()
