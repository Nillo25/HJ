#!/usr/bin/env python3
"""DXF -> standalone SVG that looks like the drawing.

    python render_dxf.py <file.dxf> <out.svg>

Coordinates are the DXF ones with Y negated, so an object stored at (x, y) is drawn at (x, -y) and the viewer can
lay its own shapes over this picture without any transform.

Drawn: LINE, LWPOLYLINE and POLYLINE (with arc segments), ARC, CIRCLE, ELLIPSE, SPLINE, SOLID, HATCH polygon
boundaries, TEXT and MTEXT (rotated), and INSERT of blocks (scaled, rotated, nested). Colours come from the
DXF (entity colour, else its layer), except the eight layers of the swiss plans which have a fixed style.
Line widths follow the size of the drawing, so a plan in millimetres is as legible as one in metres.
"""
import math
import sys
from xml.sax.saxutils import escape

# fixed style of the swiss plan layers: (fill, stroke, width factor, dash)
STYLE = {
    "ROOM_OUTLINE": ("#f6f8fb", "#cfd6e0", 0.6, None),
    "UNIT_OUTLINE": ("none", "#98a2b3", 1.6, "9 5"),
    "WALL": ("#2f3744", "#2f3744", 0.6, None),
    "WINDOW": ("#d6ecfb", "#1f7fbf", 1.0, None),
    "DOOR": ("none", "#8a5a1f", 1.0, None),
    "FIXTURE": ("none", "#6d5bb0", 0.8, None),
    "ANNOTATION": ("none", "#667085", 0.6, None),
}
ORDER = ["ROOM_OUTLINE", "UNIT_OUTLINE", "WALL", "WINDOW", "DOOR", "FIXTURE", "ANNOTATION"]
CLOSED_LAYERS = {"WALL", "WINDOW", "ROOM_OUTLINE", "UNIT_OUTLINE"}      # polylines of these layers are polygons
MAX_DEPTH = 4
MAX_ENTITIES = 400_000


# ------------------------------------------------------------------ reading
def read_pairs(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = [l.rstrip("\n") for l in f]
    return [(raw[i].strip(), raw[i + 1].strip()) for i in range(0, len(raw) - 1, 2)]


def section(pairs, name):
    """The pairs between SECTION/<name> and ENDSEC."""
    start = None
    for i, (c, v) in enumerate(pairs):
        if c == "2" and v == name and i and pairs[i - 1] == ("0", "SECTION"):
            start = i + 1
        elif start is not None and c == "0" and v == "ENDSEC":
            return pairs[start:i]
    return []


def entities(pairs):
    """[{type, code: [values]}], with POLYLINE + VERTEX folded into one entity (codes 10/20 = its vertices)."""
    out, cur, poly = [], None, None
    for c, v in pairs:
        if c == "0":
            if v == "VERTEX" and poly is not None:
                cur = {"type": "VERTEX"}
                poly.setdefault("_v", []).append(cur)
                continue
            if v == "SEQEND":
                poly, cur = None, None
                continue
            cur = {"type": v}
            out.append(cur)
            poly = cur if v == "POLYLINE" else None
        elif cur is not None:
            cur.setdefault(c, []).append(v)
    return out


def blocks_of(pairs):
    """{name: (base point, [entities])}."""
    out, name, base, buf = {}, None, (0.0, 0.0), []
    for c, v in section(pairs, "BLOCKS"):
        if c == "0" and v == "BLOCK":
            name, base, buf = None, (0.0, 0.0), []
        elif c == "0" and v == "ENDBLK":
            if name is not None:
                out[name] = (base, entities(buf))
            name = None
        elif name is None and c == "2" and buf == []:
            name = v
        else:
            if name is not None and not buf and c == "10":
                base = (float(v), base[1])
            elif name is not None and not buf and c == "20":
                base = (base[0], float(v))
            else:
                buf.append((c, v))
    return out


def layer_colours(pairs):
    out, cur = {}, None
    for c, v in section(pairs, "TABLES"):
        if c == "0":
            cur = {} if v == "LAYER" else None
            if cur is not None:
                out.setdefault("_rows", []).append(cur)
        elif cur is not None:
            cur[c] = v
    return {r["2"]: r for r in out.get("_rows", []) if "2" in r}


def aci(n):
    """AutoCAD colour index -> css. Dark for 7 (white on a black screen is black on paper)."""
    base = {1: "#d92b2b", 2: "#c9a400", 3: "#1f9d3a", 4: "#139ab8", 5: "#2b4bd9", 6: "#b02bb0", 7: "#222222",
            8: "#808080", 9: "#a0a0a0"}
    if n in base:
        return base[n]
    if 10 <= n <= 249:
        hue = ((n - 10) // 10) * 15
        light = (n - 10) % 10
        l = 0.5 - 0.03 * (light // 2)
        return f"hsl({hue},{80 - 5 * (light % 2)}%,{int(l * 100)}%)"
    return "#666666"


def colour(e, lay, layers):
    """Entity colour, else its layer's, as css (None = use the default)."""
    def one(src):
        if "420" in src:
            v = int(float(src["420"][0] if isinstance(src["420"], list) else src["420"]))
            return f"#{v & 0xFFFFFF:06x}"
        if "62" in src:
            n = int(float(src["62"][0] if isinstance(src["62"], list) else src["62"]))
            if 0 < n < 256:
                return aci(n)
        return None
    c = one(e)
    if c:
        return c
    l = layers.get(lay)
    return one(l) if l else None


# ------------------------------------------------------------------ geometry
def f(e, code, default=0.0):
    try:
        return float((e.get(code) or [default])[0])
    except (TypeError, ValueError):
        return default


class M:
    """2D affine transform: x' = a x + c y + tx, y' = b x + d y + ty."""

    def __init__(self, a=1.0, b=0.0, c=0.0, d=1.0, tx=0.0, ty=0.0):
        self.a, self.b, self.c, self.d, self.tx, self.ty = a, b, c, d, tx, ty

    def __call__(self, p):
        return (self.a * p[0] + self.c * p[1] + self.tx, self.b * p[0] + self.d * p[1] + self.ty)

    def then(self, o):
        """o after self."""
        return M(o.a * self.a + o.c * self.b, o.b * self.a + o.d * self.b,
                 o.a * self.c + o.c * self.d, o.b * self.c + o.d * self.d,
                 o.a * self.tx + o.c * self.ty + o.tx, o.b * self.tx + o.d * self.ty + o.ty)

    @property
    def scale(self):
        return math.hypot(self.a, self.b)

    @property
    def angle(self):
        return math.degrees(math.atan2(self.b, self.a))


def insert_matrix(e, base):
    sx, sy = f(e, "41", 1.0), f(e, "42", 1.0)
    r = math.radians(f(e, "50"))
    cs, sn = math.cos(r), math.sin(r)
    to_origin = M(tx=-base[0], ty=-base[1])
    return to_origin.then(M(sx * cs, sx * sn, -sy * sn, sy * cs, f(e, "10"), f(e, "20")))


def bulge_points(p, q, bulge, steps=0):
    """Points of the arc between p and q for a DXF bulge (tan of a quarter of the included angle)."""
    if abs(bulge) < 1e-9:
        return [q]
    theta = 4 * math.atan(bulge)
    chord = math.dist(p, q)
    if chord < 1e-12:
        return [q]
    r = chord / (2 * math.sin(abs(theta) / 2))
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    h = math.sqrt(max(r * r - (chord / 2) ** 2, 0)) * (1 if bulge > 0 else -1) * (1 if abs(theta) <= math.pi else -1)
    ux, uy = (q[0] - p[0]) / chord, (q[1] - p[1]) / chord
    cx, cy = mx - uy * h, my + ux * h
    a0 = math.atan2(p[1] - cy, p[0] - cx)
    n = steps or max(4, int(abs(theta) / math.radians(10)))
    return [(cx + r * math.cos(a0 + theta * i / n), cy + r * math.sin(a0 + theta * i / n)) for i in range(1, n + 1)]


def poly_vertices(e):
    """[(x, y, bulge)] of an LWPOLYLINE or a POLYLINE."""
    if e["type"] == "POLYLINE":
        return [(f(v, "10"), f(v, "20"), f(v, "42")) for v in e.get("_v", []) if int(f(v, "70")) & 16 == 0]
    xs, ys = e.get("10", []), e.get("20", [])
    bs = e.get("42", [])
    if len(bs) not in (0, len(xs)):
        bs = []                       # bulges are only listed for some vertices: cannot be matched by position
    return [(float(xs[i]), float(ys[i]), float(bs[i]) if bs else 0.0) for i in range(min(len(xs), len(ys)))]


def path_d(pts, closed):
    return "M" + "L".join(f"{x:.3f} {-y:.3f}" for x, y in pts) + ("Z" if closed else "")


# ------------------------------------------------------------------ rendering
class Canvas:
    def __init__(self, layers):
        self.layers = layers
        self.by_layer = {}            # layer -> [(kind, d or attrs, colour)]
        self.texts = []
        self.xs, self.ys = [], []
        self.n = 0

    def grow(self, pts):
        for x, y in pts:
            self.xs.append(x)
            self.ys.append(-y)

    def add(self, layer, kind, data, col, pts):
        self.by_layer.setdefault(layer, []).append((kind, data, col))
        self.grow(pts)

    def draw(self, ents, m, blocks, depth=0, parent_layer=None, parent_col=None):
        for e in ents:
            self.n += 1
            if self.n > MAX_ENTITIES:
                return
            t = e["type"]
            lay = (e.get("8") or [parent_layer or "0"])[0]
            if lay == "0" and parent_layer:
                lay = parent_layer                    # layer 0 inside a block takes the layer of the INSERT
            col = colour(e, lay, self.layers) if lay not in STYLE else None
            if col is None and parent_col and lay not in STYLE:
                col = parent_col
            if t == "INSERT":
                name = (e.get("2") or [""])[0]
                if depth < MAX_DEPTH and name in blocks:
                    base, sub = blocks[name]
                    nx = int(f(e, "70", 1)) if "70" in e else 1
                    ny = int(f(e, "71", 1)) if "71" in e else 1
                    for i in range(max(nx, 1)):
                        for j in range(max(ny, 1)):
                            em = dict(e)
                            em["10"] = [str(f(e, "10") + i * f(e, "44"))]
                            em["20"] = [str(f(e, "20") + j * f(e, "45"))]
                            self.draw(sub, insert_matrix(em, base).then(m), blocks, depth + 1, lay, col)
                continue
            if t in ("LWPOLYLINE", "POLYLINE"):
                vs = poly_vertices(e)
                if len(vs) < 2:
                    continue
                closed = bool(int(f(e, "70")) & 1)
                pts = [(vs[0][0], vs[0][1])]
                for i in range(1, len(vs) + (1 if closed else 0)):
                    a, b = vs[i - 1], vs[i % len(vs)]
                    pts += bulge_points((a[0], a[1]), (b[0], b[1]), a[2])
                if closed:
                    pts = pts[:-1] if pts[-1] == pts[0] else pts
                pts = [m(p) for p in pts]
                self.add(lay, "path", path_d(pts, closed or lay in CLOSED_LAYERS), col, pts)
            elif t == "LINE":
                pts = [m((f(e, "10"), f(e, "20"))), m((f(e, "11"), f(e, "21")))]
                self.add(lay, "path", path_d(pts, False), col, pts)
            elif t in ("ARC", "CIRCLE"):
                cx, cy, r = f(e, "10"), f(e, "20"), f(e, "40")
                if r <= 0:
                    continue
                if t == "CIRCLE":
                    pts = [m((cx + r * math.cos(a), cy + r * math.sin(a))) for a in [k * math.pi / 18 for k in range(36)]]
                    self.add(lay, "circle" if abs(m.a - m.d) < 1e-9 and abs(m.b) < 1e-9 else "path",
                             (f'cx="{m((cx, cy))[0]:.3f}" cy="{-m((cx, cy))[1]:.3f}" r="{r * m.scale:.3f}"'
                              if abs(m.a - m.d) < 1e-9 and abs(m.b) < 1e-9 else path_d(pts, True)), col, pts)
                else:
                    a0, a1 = f(e, "50"), f(e, "51")
                    sweep = (a1 - a0) % 360 or 360
                    n = max(6, int(sweep / 8))
                    pts = [m((cx + r * math.cos(math.radians(a0 + sweep * k / n)), cy + r * math.sin(math.radians(a0 + sweep * k / n))))
                           for k in range(n + 1)]
                    self.add(lay, "arc", path_d(pts, False), col, pts)
            elif t == "ELLIPSE":
                cx, cy, mx, my, ratio = f(e, "10"), f(e, "20"), f(e, "11"), f(e, "21"), f(e, "40", 1.0)
                t0, t1 = f(e, "41", 0.0), f(e, "42", 2 * math.pi)
                a = math.hypot(mx, my)
                rot = math.atan2(my, mx)
                span = (t1 - t0) % (2 * math.pi) or 2 * math.pi
                pts = []
                for k in range(65):
                    u = t0 + span * k / 64
                    x, y = a * math.cos(u), a * ratio * math.sin(u)
                    pts.append(m((cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot))))
                self.add(lay, "arc", path_d(pts, False), col, pts)
            elif t == "SPLINE":
                fx = [float(v) for v in e.get("11", [])]
                fy = [float(v) for v in e.get("21", [])]
                cxs = [float(v) for v in e.get("10", [])]
                cys = [float(v) for v in e.get("20", [])]
                src = list(zip(fx, fy)) if len(fx) >= 2 else list(zip(cxs, cys))
                if len(src) >= 2:
                    pts = [m(p) for p in src]
                    self.add(lay, "arc", path_d(pts, False), col, pts)
            elif t == "SOLID":
                q = [(f(e, "10"), f(e, "20")), (f(e, "11"), f(e, "21")), (f(e, "13"), f(e, "23")), (f(e, "12"), f(e, "22"))]
                pts = [m(p) for p in q]
                self.add(lay, "hatch", path_d(pts, True), col, pts)
            elif t == "HATCH":
                counts = [int(float(v)) for v in e.get("93", [])]
                xs = [float(v) for v in e.get("10", [])][1:]        # the first pair is the elevation point
                ys = [float(v) for v in e.get("20", [])][1:]
                i = 0
                for n in counts:
                    if n < 3 or i + n > min(len(xs), len(ys)):
                        break
                    pts = [m(p) for p in zip(xs[i:i + n], ys[i:i + n])]
                    self.add(lay, "hatch", path_d(pts, True), col, pts)
                    i += n
            elif t in ("TEXT", "MTEXT"):
                txt = (e.get("1") or [""])[0]
                if t == "MTEXT":
                    txt = txt.replace("\\P", " ").replace("\\~", " ")
                    for tag in ("{", "}"):
                        txt = txt.replace(tag, "")
                if txt.strip():
                    x, y = m((f(e, "10"), f(e, "20")))
                    ang = f(e, "50") + m.angle
                    size = max(f(e, "40", 0.25), 1e-6) * m.scale
                    self.texts.append((x, y, size, txt.strip(), lay, col, ang, int(f(e, "72")) if t == "TEXT" else 0))

    def svg(self):
        if not self.xs:
            raise ValueError("nothing to draw")
        # robust extent: a stray point far away must not shrink the drawing to a dot
        xs, ys = sorted(self.xs), sorted(self.ys)
        lo, hi = int(len(xs) * 0.002), max(int(len(xs) * 0.998) - 1, 0)
        x0, x1, y0, y1 = xs[lo], xs[hi], ys[lo], ys[hi]
        size = max(x1 - x0, y1 - y0, 1e-6)
        pad = size * 0.02
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        w, h = x1 - x0, y1 - y0
        unit = size / 1600                                          # one hairline in drawing units
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.3f} {y0:.3f} {w:.3f} {h:.3f}">',
               f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{w:.3f}" height="{h:.3f}" fill="#ffffff"/>']
        for lay in ORDER + sorted(set(self.by_layer) - set(ORDER)):
            items = self.by_layer.get(lay)
            if not items:
                continue
            fill, stroke, wf, dash = STYLE.get(lay, ("none", "#444444", 1.0, None))
            for kind, data, col in items:
                s = col or stroke
                da = f' stroke-dasharray="{dash}"' if dash else ""
                sw = f'stroke-width="{unit * wf:.4f}" stroke-linejoin="round"'
                if kind == "path":
                    out.append(f'<path d="{data}" fill="{fill if lay in STYLE else "none"}" stroke="{s}" {sw}{da}/>')
                elif kind == "hatch":
                    out.append(f'<path d="{data}" fill="{col or ("#2f3744" if lay in STYLE else "#9aa3b0")}" fill-opacity="{1 if lay in STYLE else 0.5}" stroke="none"/>')
                elif kind == "arc":
                    out.append(f'<path d="{data}" fill="none" stroke="{s}" stroke-opacity="0.85" stroke-width="{unit * wf * 0.7:.4f}"/>')
                else:
                    out.append(f'<circle {data} fill="none" stroke="{s}" {sw}/>')
        out.append('<g font-family="Arial, Helvetica, sans-serif">')
        for x, y, size, txt, lay, col, ang, just in self.texts:
            c = col or ("#667085" if lay == "ANNOTATION" else "#1f2937")
            anchor = "middle" if just in (1, 4) else "end" if just in (2, 3, 5) else "start"
            rot = f' transform="rotate({-ang:.2f} {x:.3f} {-y:.3f})"' if abs(ang) > 0.01 else ""
            out.append(f'<text x="{x:.3f}" y="{-y + size * 0.35:.3f}" font-size="{size:.3f}" fill="{c}" text-anchor="{anchor}"{rot}>{escape(txt)}</text>')
        out.append("</g></svg>")
        return "\n".join(out)


def render(path):
    pairs = read_pairs(path)
    layers = layer_colours(pairs)
    blocks = blocks_of(pairs)
    cv = Canvas(layers)
    cv.draw(entities(section(pairs, "ENTITIES")), M(), blocks)
    return cv.svg()


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: render_dxf.py file.dxf out.svg")
    svg = render(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"{len(svg) // 1024} KB")


if __name__ == "__main__":
    main()
