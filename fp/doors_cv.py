"""Двери по дугам открывания — без нейросети, для любого листа.

На любом чертеже распашная дверь — это створка (прямая от петли длиной в ширину двери) и четверть окружности,
которую описывает её край. Ищем такие пары свёрткой с шаблонами:
- дуга: доля точек четверти окружности радиуса r, лежащих на тёмной линии;
- створка: доля точек одного из двух радиусов, ограничивающих четверть, лежащих на линии;
- пусто: внутри сектора (пол, по которому ходит дверь) почти нет линий;
- не круг: противоположная четверть почти пустая (кружки марок, оборудование, отводы труб дают полную окружность);
- тонкая дуга: на соседних радиусах r ± RING линии нет — иначе отклик даёт густое место (текст, трубы, штриховка).
Отводы труб — дуги без радиальной створки, кружки марок — с текстом внутри, их отсекают условия выше.
Две зеркальные дуги с общим проёмом — одна двустворчатая дверь.

Радиус — ширина двери 0.6–1.3 м по масштабу листа; без масштаба — по размеру дверей, найденных моделью.
"""
from __future__ import annotations

import cv2
import numpy as np

DOOR_MIN_M, DOOR_MAX_M = 0.6, 1.3
WORK_PPM = 80.0           # крупнее — уменьшаем до ~80 px на метр; мельче не уменьшаем: дуги бывают в 1 px и бледные
DARK = 240                # порог «линии»: дуги на чертёжных листах серые (~220), ложные отклики отсекает контраст кольца
RING = 4                  # дуга тонкая: на радиусах r ± RING px линии быть не должно (иначе это густая штриховка)

# четверть задаётся направлениями двух ограничивающих радиусов (единичные векторы осей)
QUADS = [((1, 0), (0, -1)), ((0, -1), (-1, 0)), ((-1, 0), (0, 1)), ((0, 1), (1, 0))]


def _kernel(r: int, pts: np.ndarray) -> np.ndarray:
    k = np.zeros((2 * r + 5, 2 * r + 5), np.float32)
    c = r + 2
    for x, y in pts:
        k[int(round(c + y)), int(round(c + x))] = 1
    s = k.sum()
    return k / s if s else k


def _arc_pts(r: int, a: tuple, b: tuple, margin: float = 0.12) -> np.ndarray:
    """Точки дуги от направления a к b (четверть), без краёв (там дуга сливается со стеной и створкой)."""
    a0, a1 = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    ts = a0 + d * np.linspace(margin, 1 - margin, max(12, int(abs(d) * r)))
    return np.stack([r * np.cos(ts), r * np.sin(ts)], 1)


def _ray_pts(r: int, v: tuple, margin: float = 0.15) -> np.ndarray:
    ts = np.linspace(margin * r, (1 - margin) * r, max(8, r))
    return np.stack([ts * v[0], ts * v[1]], 1)


def _sector_pts(r: int, a: tuple, b: tuple) -> np.ndarray:
    pts = []
    for rr in np.linspace(0.25 * r, 0.8 * r, max(4, r // 4)):
        pts.append(_arc_pts(int(rr), a, b, margin=0.2) * (rr / int(rr) if int(rr) else 1))
    return np.concatenate(pts)


def find_door_arcs(gray: np.ndarray, px_per_m: float | None, door_px: float | None = None,
                   region=None) -> list[dict]:
    """Двери по дугам. px_per_m — масштаб листа; door_px — типичная ширина двери в px (если масштаба нет).
    region — [x0,y0,x1,y1], где искать (рамка чертежа). Возвращает [{"bbox","hinge","r_m","score","leaves"}]
    в пикселях исходной картинки; двустворчатая дверь — одна запись с leaves=2."""
    if px_per_m:
        rmin, rmax, ppm = DOOR_MIN_M * px_per_m, DOOR_MAX_M * px_per_m, px_per_m
    elif door_px:
        rmin, rmax, ppm = 0.65 * door_px, 1.45 * door_px, door_px / 0.9
    else:
        return []
    s = min(1.0, WORK_PPM / ppm)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    dark = (g < DARK).astype(np.float32)
    near = cv2.dilate(dark, np.ones((3, 3), np.uint8))     # допуск ±1 px для дуги и створки
    h, w = g.shape
    found = []
    radii = sorted({int(round(v)) for v in np.arange(rmin * s, rmax * s + 1, 3.0)})
    for r in radii:
        if r < 6 + RING:
            continue
        ray = {v: cv2.filter2D(near, -1, _kernel(r, _ray_pts(r, v)), borderType=cv2.BORDER_CONSTANT)
               for v in ((1, 0), (0, -1), (-1, 0), (0, 1))}
        def arc_resp(rr):
            return [cv2.filter2D(near, -1, _kernel(rr, _arc_pts(rr, a, b)), borderType=cv2.BORDER_CONSTANT)
                    for a, b in QUADS]
        arcs, inner, outer = arc_resp(r), arc_resp(r - RING), arc_resp(r + RING)
        for qi, (a, b) in enumerate(QUADS):
            arc = arcs[qi]
            opp = arcs[(qi + 2) % 4]
            leaf = np.maximum(ray[a], ray[b])
            ring = np.maximum(inner[qi], outer[qi])
            m = (arc >= 0.8) & (arc - ring >= 0.5) & (leaf >= 0.75) & (opp <= 0.45)
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            inside = cv2.filter2D(dark, -1, _kernel(r, _sector_pts(r, a, b)), borderType=cv2.BORDER_CONSTANT)
            for y, x in zip(ys, xs):
                if inside[y, x] > 0.08:
                    continue
                la = ray[a][y, x] >= ray[b][y, x]
                found.append({"x": x, "y": y, "r": r, "q": qi, "leaf": a if la else b,
                              "score": float(arc[y, x] + leaf[y, x] - inside[y, x])})
    # NMS: одна дуга даёт пятно откликов соседних центров и радиусов
    found.sort(key=lambda d: -d["score"])
    keep = []
    for d in found:
        if all(np.hypot(d["x"] - k["x"], d["y"] - k["y"]) > 0.5 * max(d["r"], k["r"]) for k in keep):
            keep.append(d)
    # в исходные пиксели + рамка сектора
    out = []
    for d in keep:
        a, b = QUADS[d["q"]]
        cx, cy, r = d["x"] / s, d["y"] / s, d["r"] / s
        xs_ = [cx, cx + r * a[0], cx + r * b[0]]
        ys_ = [cy, cy + r * a[1], cy + r * b[1]]
        bb = [min(xs_), min(ys_), max(xs_), max(ys_)]
        if region and not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
            continue
        out.append({"bbox": [round(v, 1) for v in bb], "hinge": [round(cx, 1), round(cy, 1)],
                    "r_m": round(r / ppm, 2), "score": round(d["score"], 2), "leaves": 1})
    return _merge_double(out, ppm)


def _merge_double(arcs: list[dict], ppm: float) -> list[dict]:
    """Двустворчатая дверь: две дуги, чьи сектора касаются по проёму, — одна дверь."""
    gap = 0.15 * ppm
    parent = list(range(len(arcs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, p in enumerate(arcs):
        for j in range(i + 1, len(arcs)):
            q = arcs[j]
            a, b = p["bbox"], q["bbox"]
            touch = a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]
            # петли на одной линии (по x или по y) на расстоянии ~ двух створок
            hx, hy = abs(p["hinge"][0] - q["hinge"][0]), abs(p["hinge"][1] - q["hinge"][1])
            r = (p["r_m"] + q["r_m"]) * ppm / 2
            aligned = (hy < 0.2 * ppm and 1.5 * r < hx < 2.5 * r) or (hx < 0.2 * ppm and 1.5 * r < hy < 2.5 * r)
            if touch and aligned:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i, d in enumerate(arcs):
        groups.setdefault(find(i), []).append(d)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        bb = [min(d["bbox"][0] for d in g), min(d["bbox"][1] for d in g),
              max(d["bbox"][2] for d in g), max(d["bbox"][3] for d in g)]
        out.append({"bbox": bb, "hinge": g[0]["hinge"], "r_m": max(d["r_m"] for d in g),
                    "score": max(d["score"] for d in g), "leaves": len(g)})
    return out


def near_walls(arcs: list[dict], walls, px_per_m: float, max_m: float = 0.5) -> list[dict]:
    """Дверь здания стоит в стене: петля не дальше max_m от границы помещения или стены.
    Дверцы оборудования, отводы и арматура посреди помещения отсекаются. walls — shapely-геометрии
    (контуры помещений и полигоны стен модели)."""
    from shapely.geometry import Point
    from shapely.ops import unary_union
    if not walls:
        return arcs
    edges = unary_union([g.boundary if g.geom_type in ("Polygon", "MultiPolygon") else g for g in walls])
    return [a for a in arcs if edges.distance(Point(*a["hinge"])) <= max_m * px_per_m]


def merge_with_model(arcs: list[dict], doors: list, px_per_m: float) -> tuple[list[dict], set]:
    """Дуги, совпавшие с дверью модели (рамки пересекаются с допуском 0.2 м), — та же дверь.
    Возвращает (дуги без пары — новые двери, id дверей модели, подтверждённых дугой)."""
    pad = 0.2 * px_per_m
    confirmed, new = set(), []
    for a in arcs:
        x0, y0, x1, y1 = a["bbox"]
        hit = None
        for d in doors:
            xs = [p[0] for p in d.polygon]
            ys = [p[1] for p in d.polygon]
            if min(xs) - pad <= x1 and x0 <= max(xs) + pad and min(ys) - pad <= y1 and y0 <= max(ys) + pad:
                hit = d.id
                break
        if hit:
            confirmed.add(hit)
        else:
            new.append(a)
    return new, confirmed
