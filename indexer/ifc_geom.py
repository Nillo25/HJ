"""2D footprints of IFC elements for the plan view. No geometry kernel.

For each product we find its Body (or FootPrint) representation and turn it into
one polygon in world XY, in metres:

  IfcExtrudedAreaSolid   profile outline (rectangle, circle, I-shape, closed
                         polyline) moved by the solid position and the element
                         placement chain. If the extrusion is not vertical
                         (a wall profile drawn in elevation) the hull of the
                         profile and its extruded copy is used.
  IfcMappedItem          the mapped representation under the item transform
  IfcBoolean*Result      the first operand (cuts are ignored)
  anything else          convex hull of every cartesian point reachable from the item
                         (faceted breps, face sets). Rough, but it shows where the element is.

Elements whose geometry is none of these get no polygon: the viewer says how many
were drawn out of how many, it never invents a shape.
"""
import re

NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
REF = re.compile(r"#(\d+)")

# classes the geometry needs (the reader keeps only wanted classes)
GEOM_KEEP = {
    "IFCLOCALPLACEMENT", "IFCAXIS2PLACEMENT3D", "IFCAXIS2PLACEMENT2D", "IFCDIRECTION",
    "IFCEXTRUDEDAREASOLID", "IFCMAPPEDITEM", "IFCREPRESENTATIONMAP",
    "IFCBOOLEANCLIPPINGRESULT", "IFCBOOLEANRESULT",
    "IFCRECTANGLEPROFILEDEF", "IFCROUNDEDRECTANGLEPROFILEDEF", "IFCCIRCLEPROFILEDEF",
    "IFCISHAPEPROFILEDEF", "IFCARBITRARYCLOSEDPROFILEDEF", "IFCARBITRARYPROFILEDEFWITHVOIDS",
    "IFCCARTESIANTRANSFORMATIONOPERATOR3D", "IFCCARTESIANTRANSFORMATIONOPERATOR2D",
    "IFCCARTESIANTRANSFORMATIONOPERATOR3DNONUNIFORM",
    "IFCFACETEDBREP", "IFCFACETEDBREPWITHVOIDS", "IFCCLOSEDSHELL", "IFCOPENSHELL", "IFCFACE",
    "IFCFACEOUTERBOUND", "IFCFACEBOUND", "IFCPOLYLOOP", "IFCSHELLBASEDSURFACEMODEL",
    "IFCTRIANGULATEDFACESET", "IFCPOLYGONALFACESET", "IFCCARTESIANPOINTLIST3D",
}

IDENT = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _refs(s):
    return [int(x) for x in REF.findall(s or "")]


def _ref(a):
    m = REF.match((a or "").strip())
    return int(m.group(1)) if m else None


def _nums(s):
    return [float(x) for x in NUM.findall(s or "")]


def _unit(v):
    n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-12 else None


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


# A matrix is (R, t): R three column vectors (x, y, z axes), t translation.
def _mul(A, B):
    """A after B: apply B first, then A."""
    (ax, ay, az), at = A
    (bx, by, bz), bt = B

    def rot(v):
        return tuple(ax[i] * v[0] + ay[i] * v[1] + az[i] * v[2] for i in range(3))

    return (rot(bx), rot(by), rot(bz)), tuple(rot(bt)[i] + at[i] for i in range(3))


def _apply(M, p):
    (x, y, z), t = M
    px, py, pz = (p[0], p[1], p[2] if len(p) > 2 else 0.0)
    return tuple(x[i] * px + y[i] * py + z[i] * pz + t[i] for i in range(3))


def _inverse(M):
    """Inverse of a rigid transform: R^T and -R^T t."""
    (x, y, z), t = M
    cols = tuple((x[k], y[k], z[k]) for k in range(3))
    dot = lambda u: sum(u[i] * t[i] for i in range(3))
    return cols, (-dot(x), -dot(y), -dot(z))


class Geom:
    def __init__(self, ents, scale):
        self.e = ents
        self.scale = scale
        self.cache = {}

    # --- primitives --------------------------------------------------------
    def point(self, ref):
        c = self.e.get(ref)
        if not c or c[0] != "IFCCARTESIANPOINT":
            return None
        n = _nums(",".join(c[1]))
        return tuple(n) if len(n) >= 2 else None

    def direction(self, ref):
        c = self.e.get(ref) if ref else None
        if not c or c[0] != "IFCDIRECTION":
            return None
        n = _nums(",".join(c[1]))
        n = (n + [0.0, 0.0, 0.0])[:3] if len(n) < 3 else n[:3]
        return _unit(tuple(n))

    def axis2(self, ref):
        """IfcAxis2Placement3D / 2D -> matrix."""
        c = self.e.get(ref) if ref else None
        if not c:
            return IDENT, (0.0, 0.0, 0.0)
        cls, a = c
        loc = self.point(_ref(a[0])) if a else None
        loc = (tuple(loc) + (0.0,))[:3] if loc and len(loc) == 2 else (loc or (0.0, 0.0, 0.0))
        if cls == "IFCAXIS2PLACEMENT2D":
            xd = self.direction(_ref(a[1])) if len(a) > 1 else None
            x = xd or (1.0, 0.0, 0.0)
            return (x, (-x[1], x[0], 0.0), (0.0, 0.0, 1.0)), loc
        z = (self.direction(_ref(a[1])) if len(a) > 1 else None) or (0.0, 0.0, 1.0)
        xr = (self.direction(_ref(a[2])) if len(a) > 2 else None) or (1.0, 0.0, 0.0)
        d = sum(xr[i] * z[i] for i in range(3))
        x = _unit(tuple(xr[i] - d * z[i] for i in range(3))) or (1.0, 0.0, 0.0)
        return (x, _cross(z, x), z), loc

    def placement(self, ref, depth=0):
        """IfcLocalPlacement chain -> world matrix."""
        if ref is None or depth > 40:
            return IDENT, (0.0, 0.0, 0.0)
        if ref in self.cache:
            return self.cache[ref]
        c = self.e.get(ref)
        if not c or c[0] != "IFCLOCALPLACEMENT":
            return IDENT, (0.0, 0.0, 0.0)
        a = c[1]
        parent = self.placement(_ref(a[0]), depth + 1) if a and _ref(a[0]) else (IDENT, (0.0, 0.0, 0.0))
        m = _mul(parent, self.axis2(_ref(a[1])) if len(a) > 1 else (IDENT, (0.0, 0.0, 0.0)))
        self.cache[ref] = m
        return m

    def operator(self, ref):
        """IfcCartesianTransformationOperator -> matrix."""
        c = self.e.get(ref) if ref else None
        if not c:
            return IDENT, (0.0, 0.0, 0.0)
        cls, a = c
        x = (self.direction(_ref(a[0])) if a else None) or (1.0, 0.0, 0.0)
        y = (self.direction(_ref(a[1])) if len(a) > 1 else None) or (0.0, 1.0, 0.0)
        o = self.point(_ref(a[2])) if len(a) > 2 else None
        o = (tuple(o) + (0.0,))[:3] if o and len(o) == 2 else (o or (0.0, 0.0, 0.0))
        s = _nums(a[3])[0] if len(a) > 3 and _nums(a[3]) else 1.0
        z = (self.direction(_ref(a[4])) if len(a) > 4 else None) or _cross(x, y)
        sc = lambda v: (v[0] * s, v[1] * s, v[2] * s)
        return (sc(x), sc(y), sc(z)), o

    # --- profiles ----------------------------------------------------------
    def profile(self, ref):
        """Profile outline in profile coordinates, or None."""
        c = self.e.get(ref) if ref else None
        if not c:
            return None
        cls, a = c
        pos = self.axis2(_ref(a[2])) if len(a) > 2 and _ref(a[2]) else (IDENT, (0.0, 0.0, 0.0))
        pts = None
        if cls in ("IFCRECTANGLEPROFILEDEF", "IFCROUNDEDRECTANGLEPROFILEDEF") and len(a) > 4:
            w, h = (_nums(a[3]) or [0])[0], (_nums(a[4]) or [0])[0]
            pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        elif cls == "IFCISHAPEPROFILEDEF" and len(a) > 4:
            w, h = (_nums(a[3]) or [0])[0], (_nums(a[4]) or [0])[0]
            pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        elif cls == "IFCCIRCLEPROFILEDEF" and len(a) > 3:
            r = (_nums(a[3]) or [0])[0]
            import math
            pts = [(r * math.cos(i * math.pi / 8), r * math.sin(i * math.pi / 8)) for i in range(16)]
        elif cls in ("IFCARBITRARYCLOSEDPROFILEDEF", "IFCARBITRARYPROFILEDEFWITHVOIDS") and len(a) > 2:
            curve = self.e.get(_ref(a[2]))
            if curve and curve[0] == "IFCPOLYLINE":
                pts = [self.point(r) for r in _refs(",".join(curve[1]))]
                pts = [p[:2] for p in pts if p]
        if not pts or max(abs(v) for p in pts for v in p) == 0:
            return None
        if cls.startswith("IFCARBITRARY"):
            return pts                                   # outer curve is already in profile coordinates
        return [_apply(pos, (p[0], p[1], 0.0))[:2] for p in pts]

    # --- items -------------------------------------------------------------
    def reach_points(self, ref, M, cap=20000):
        """Every cartesian point under an item, transformed by M (3D)."""
        out, seen, stack = [], set(), [ref]
        while stack and len(out) < cap:
            r = stack.pop()
            if r in seen or r not in self.e:
                continue
            seen.add(r)
            cls, a = self.e[r]
            if cls == "IFCCARTESIANPOINT":
                p = self.point(r)
                if p:
                    out.append(_apply(M, p))
                continue
            if cls == "IFCCARTESIANPOINTLIST3D":
                n = _nums(",".join(a))
                out += [_apply(M, tuple(n[i:i + 3])) for i in range(0, len(n) - 2, 3)]
                continue
            stack += _refs(",".join(a))
        return out

    def item(self, ref, M, depth=0):
        """-> list of world polygons ([(x, y), ...]) for one representation item."""
        c = self.e.get(ref)
        if not c or depth > 8:
            return []
        cls, a = c
        if cls == "IFCEXTRUDEDAREASOLID" and len(a) > 3:
            prof = self.profile(_ref(a[0]))
            if not prof:
                return []
            pos = _mul(M, self.axis2(_ref(a[1])) if _ref(a[1]) else (IDENT, (0.0, 0.0, 0.0)))
            d = self.direction(_ref(a[2])) or (0.0, 0.0, 1.0)
            depth_v = (_nums(a[3]) or [0.0])[0]
            base = [_apply(pos, (p[0], p[1], 0.0)) for p in prof]
            ex = _apply(pos, tuple(d[i] * depth_v for i in range(3)))
            top_dz = ex[2] - _apply(pos, (0, 0, 0))[2]
            ext_len = depth_v or 1.0
            if abs(top_dz) / ext_len > 0.9:                 # vertical extrusion: the outline is exact
                return [[(p[0], p[1]) for p in base]]
            off = tuple(ex[i] - _apply(pos, (0, 0, 0))[i] for i in range(3))
            return [_hull([(p[0], p[1]) for p in base] + [(p[0] + off[0], p[1] + off[1]) for p in base])]
        if cls == "IFCMAPPEDITEM" and len(a) > 1:
            src = self.e.get(_ref(a[0]))
            if not src or src[0] != "IFCREPRESENTATIONMAP":
                return []
            origin = self.axis2(_ref(src[1][0])) if _ref(src[1][0]) else (IDENT, (0.0, 0.0, 0.0))
            tgt = self.operator(_ref(a[1]))
            M2 = _mul(M, _mul(tgt, _inverse(origin)))
            rep = self.e.get(_ref(src[1][1])) if len(src[1]) > 1 else None
            out = []
            if rep:
                for it in _refs(rep[1][3] if len(rep[1]) > 3 else ""):
                    out += self.item(it, M2, depth + 1)
            return out
        if cls in ("IFCBOOLEANCLIPPINGRESULT", "IFCBOOLEANRESULT") and len(a) > 1:
            return self.item(_ref(a[1]), M, depth + 1)
        if cls == "IFCPOLYLINE":
            pts = [self.point(r) for r in _refs(",".join(a))]
            pts = [_apply(M, p) for p in pts if p]
            return [[(p[0], p[1]) for p in pts]] if len(pts) >= 3 else []
        pts = self.reach_points(ref, M)
        if len(pts) >= 3:
            return [_hull([(p[0], p[1]) for p in pts])]
        return []

    # --- element -----------------------------------------------------------
    def footprint(self, args, want="Body"):
        """Polygon in metres for a product with ObjectPlacement at 5 and Representation at 6."""
        if len(args) < 7:
            return None
        pl, rep = _ref(args[5]), _ref(args[6])
        if not rep or rep not in self.e or self.e[rep][0] != "IFCPRODUCTDEFINITIONSHAPE":
            return None
        M = self.placement(pl)
        reps = [self.e[r] for r in _refs(self.e[rep][1][2] if len(self.e[rep][1]) > 2 else "")
                if r in self.e and self.e[r][0] == "IFCSHAPEREPRESENTATION"]

        def ident(r):
            return (r[1][1] or "").strip("'") if len(r[1]) > 1 else ""

        order = [want, "Body", "FootPrint", "Box"] if want == "Body" else [want, "Body"]
        chosen = None
        for w in order:
            chosen = next((r for r in reps if ident(r) == w), None)
            if chosen:
                break
        if not chosen:
            return None
        polys = []
        for it in _refs(chosen[1][3] if len(chosen[1]) > 3 else ""):
            polys += self.item(it, M)
        polys = [p for p in polys if len(p) >= 3]
        if not polys:
            return None
        pts = polys[0] if len(polys) == 1 else _hull([q for p in polys for q in p])
        s = self.scale
        return [(round(x * s, 3), round(y * s, 3)) for x, y in pts]


def _hull(points):
    """Convex hull (monotone chain)."""
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]
