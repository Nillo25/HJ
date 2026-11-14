"""Cluster DXF primitives into objects; classify fixtures.

A door or a fixture is drawn as a dozen segments, arcs and circles.
Nothing can be counted or classified until primitives are clustered.
This fixes both weak spots at once: doors and fixture subtypes.

Classification uses bounding box size plus the room the object sits in.
Thresholds are tuned on European plans (swiss_dwellings); other drawing
conventions need retuning, and we say so out loud.
"""
import math

# ------------------------------------------------------------- geometry

def bbox_of(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def bbox_gap(a, b) -> float:
    """Gap between boxes. 0 means they overlap or touch."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def dims(bb):
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    lo, hi = min(w, h), max(w, h)
    return w, h, hi, lo, (hi / lo if lo > 1e-9 else 999.0)


# ----------------------------------------------------------- clustering

def cluster(prims, gap=0.05):
    """prims: [{'bbox': [...], ...}]. Merges anything closer than gap metres.

    Plain pairwise union-find. A layer holds a few hundred primitives, so
    quadratic cost is fine here.
    """
    n = len(prims)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    for i in range(n):
        for j in range(i + 1, n):
            if bbox_gap(prims[i]["bbox"], prims[j]["bbox"]) <= gap:
                union(i, j)

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(prims[i])

    out = []
    for members in groups.values():
        bb = members[0]["bbox"]
        for m in members[1:]:
            bb = bbox_union(bb, m["bbox"])
        out.append({"bbox": bb, "members": members, "n": len(members),
                    "has_circle": any(m.get("shape") == "circle" for m in members),
                    "has_arc": any(m.get("shape") == "arc" for m in members)})
    return out


# ------------------------------------------- fixture classification
# Real-world sizes in metres. Bands are deliberately wide: better to fall
# through to "other" than to call a bathtub a shower.

def classify_fixture(group, room_subtype: str | None) -> tuple[str | None, float]:
    """-> (subtype, confidence). None means unrecognised -> fixture/other."""
    _, _, long, short, aspect = dims(group["bbox"])
    area = long * short

    # bathtub: long and narrow
    if 1.4 <= long <= 2.0 and 0.6 <= short <= 0.9:
        return "bathtub", 0.9

    # shower tray: near square, clearly smaller than a bathtub
    if 0.7 <= long <= 1.2 and aspect <= 1.35:
        return "shower", 0.85

    # toilet: small and elongated; never in a kitchen
    if 0.5 <= long <= 0.85 and 0.3 <= short <= 0.5 and room_subtype != "kitchen":
        if room_subtype in ("wc", "bathroom", None):
            return "toilet", 0.85 if room_subtype else 0.6

    # washbasin / sink: small, often a circle
    if 0.3 <= long <= 0.9 and short <= 0.65:
        # the room decides the hard pair: toilet vs washbasin
        if room_subtype == "kitchen":
            return "sink", 0.9
        if group["has_circle"]:
            return "sink", 0.8
        return "sink", 0.55

    return None, 0.3


# ------------------------------------------------ doors from primitives

def classify_door(group) -> tuple[str, float]:
    """Door type from a clustered group.

    The naive "one arc = one door" rule scored 56%: sliding doors have no
    arc and double doors have two. Count groups; arcs inside a group give
    the subtype.
    """
    arcs = sum(1 for m in group["members"] if m.get("shape") == "arc")
    _, _, long, _, _ = dims(group["bbox"])
    if arcs >= 2:
        return "double", 0.9
    if arcs == 1:
        return "single", 0.9
    if long >= 0.6:
        return "sliding", 0.6      # leaf without an arc
    return "opening", 0.5          # opening with no leaf - NOT a door


# ------------------------------------------------------------ diagnostics

def histogram(groups, bins=(0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0)):
    """Size distribution - look at this BEFORE picking thresholds.

    If sizes fall into clear bands, thresholds are obvious.
    If they smear, clustering is broken - fix that, not the thresholds.
    """
    rows = []
    for g in groups:
        _, _, long, short, aspect = dims(g["bbox"])
        rows.append((round(long, 2), round(short, 2), round(aspect, 2), g["n"]))
    rows.sort(reverse=True)
    counts = {b: 0 for b in bins}
    for long, *_ in rows:
        for b in bins:
            if long <= b:
                counts[b] += 1
                break
    return rows, counts
