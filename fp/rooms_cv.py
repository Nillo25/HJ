"""Помещения по стенам — без нейросети, для любого листа (модель YOLO обучена только на квартирах CubiCasa).

1. Стены = толстые тёмные штрихи: бинаризация + открытие (убирает текст, мебель, размерные линии).
2. Дверные проёмы закрываем: от каждого «торца» стены луч вдоль стены до следующей стены на расстоянии
   0.5–1.6 м (ширина проёма). Закрытые отрезки запоминаем — это проёмы (для плинтусов).
3. Помещения = связные области свободного пола, не касающиеся края листа.
4. Площадь = площадь области (пол между стенами, как в подписях «10,6 m²»);
   плинтус = периметр области минус ширина проёмов этого помещения.
"""
from __future__ import annotations

import cv2
import numpy as np

DOOR_MIN_M, DOOR_MAX_M = 0.5, 1.6       # дверной проём: луч от торца до любой стены
WINDOW_MAX_M = 3.5                     # проём в той же линии стены (окно): луч до соосного торца
ROOM_MIN_M2, ROOM_MAX_M2 = 0.8, 400.0


def wall_mask(gray: np.ndarray, thick: int | None = None, k: int | None = None) -> tuple[np.ndarray, int]:
    dark = (gray < 140).astype(np.uint8)
    if thick is None:
        # толщина стены: типичная ширина толстых штрихов (2 × расстояние до края в «хребтах» штрихов)
        dt = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
        ridge = dt[(dt >= 2) & (dt == cv2.dilate(dt, np.ones((3, 3), np.uint8)))]
        thick = int(round(2 * np.percentile(ridge, 75))) if ridge.size > 100 else 6
        thick = int(np.clip(thick, 3, 40))
    k = k or max(2, int(thick * 0.8))
    walls = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return walls, thick


def _endpoints(walls: np.ndarray) -> list[tuple[int, int, float, float]]:
    """Торцы стен: точки скелета с одним соседом и направление «наружу» вдоль стены."""
    from skimage.morphology import skeletonize
    sk = skeletonize(walls > 0).astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32)) - sk
    ys, xs = np.nonzero((sk == 1) & (nb == 1))
    out = []
    for y, x in zip(ys, xs):
        # идём по скелету назад на ~8 px, направление = от той точки к торцу
        py, px_, cy, cx = -1, -1, y, x
        for _ in range(8):
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = cy + dy, cx + dx
                    if (dy or dx) and 0 <= yy < sk.shape[0] and 0 <= xx < sk.shape[1] and sk[yy, xx] \
                            and (yy, xx) != (py, px_):
                        nxt = (yy, xx)
            if not nxt:
                break
            py, px_, (cy, cx) = cy, cx, nxt
        vx, vy = x - cx, y - cy
        n = np.hypot(vx, vy)
        if n >= 3:
            out.append((int(x), int(y), vx / n, vy / n))
    return out


def close_openings(walls: np.ndarray, thick: int, px_per_m: float | None) -> tuple[np.ndarray, list]:
    """Луч от торца стены вдоль её направления.
    Упёрся в любую стену на 0.5–1.6 м — дверной проём; упёрся в соосный встречный торец до 3.5 м — окно/проём
    в той же стене. Закрываем отрезком, запоминаем ширину (для плинтуса)."""
    h, w = walls.shape
    ppm = px_per_m or thick / 0.15  # без масштаба: стена ~15 см
    dmin, dmax, wmax = int(DOOR_MIN_M * ppm), int(DOOR_MAX_M * ppm), int(WINDOW_MAX_M * ppm)
    ends = []
    for x, y, vx, vy in _endpoints(walls):
        if abs(vx) > abs(vy):  # стены почти всегда горизонтальны/вертикальны
            ends.append((x, y, float(np.sign(vx)), 0.0))
        else:
            ends.append((x, y, 0.0, float(np.sign(vy))))
    closed = walls.copy()
    gaps, seen = [], set()
    for i, (x, y, vx, vy) in enumerate(ends):
        hit = None
        for d in range(thick, wmax + 1):
            xx, yy = int(round(x + vx * d)), int(round(y + vy * d))
            if not (0 <= xx < w and 0 <= yy < h):
                break
            if walls[yy, xx]:
                hit = (xx, yy, d)
                break
        if not hit:
            continue
        xx, yy, d = hit
        if d < dmin:
            continue
        # встречный соосный торец рядом с точкой попадания?
        partner = next((j for j, (ex, ey, evx, evy) in enumerate(ends)
                        if j != i and evx == -vx and evy == -vy and abs(ex - xx) + abs(ey - yy) <= 2 * thick
                        and (abs(ey - y) <= thick if vx else abs(ex - x) <= thick)), None)
        if d > dmax and partner is None:
            continue  # длинный луч через комнату к боковой стене — не проём
        key = tuple(sorted((i, partner))) if partner is not None else (i,)
        if key in seen:
            continue
        seen.add(key)
        cv2.line(closed, (x, y), (xx, yy), 1, thickness=max(2, thick // 2))
        gaps.append(((x, y), (xx, yy), float(d)))
    return closed, gaps


def find_spaces(img: np.ndarray, px_per_m: float | None = None) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    walls, thick = wall_mask(gray)
    closed, gaps = close_openings(walls, thick, px_per_m)
    free = (closed == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    h, w = gray.shape
    ppm = px_per_m
    spaces = []
    gap_mask_ids = []
    for g in gaps:  # проём -> маска отрезка для поиска соседних помещений
        m = np.zeros((h, w), np.uint8)
        cv2.line(m, g[0], g[1], 1, thickness=thick + 4)
        gap_mask_ids.append((m.astype(bool), g[2]))
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue  # снаружи здания
        if ppm:
            m2 = area / ppm ** 2
            if not ROOM_MIN_M2 <= m2 <= ROOM_MAX_M2:
                continue
        elif area < 0.0005 * h * w:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        perim = cv2.arcLength(c, True)
        near = cv2.dilate(comp, np.ones((thick + 6, thick + 6), np.uint8)).astype(bool)
        openings = [d for gm, d in gap_mask_ids if (gm & near).any()]
        # отношение площади к площади рамки: узкие «щели» между двойными линиями стен — не помещения
        if area / (bw * bh) < 0.25 or min(bw, bh) < 2 * thick:
            continue
        spaces.append({"polygon": poly.tolist(), "area_px2": float(area), "perimeter_px": float(perim),
                       "openings_px": openings,
                       "baseboard_px": float(max(0.0, perim - sum(openings)))})
    return {"spaces": spaces, "wall_thickness_px": thick, "openings": len(gaps)}


# ---------------- заливка от подписи комнаты (основной способ) ----------------
def barrier_mask(gray: np.ndarray) -> np.ndarray:
    """Всё тёмное — преграда (стены из двух тонких линий, окна, дуги дверей перекрывают проёмы).
    Лёгкое утолщение закрывает разрывы сканирования в 1–2 px."""
    dark = (gray < 170).astype(np.uint8)
    return cv2.dilate(dark, np.ones((3, 3), np.uint8))


def _flood(barrier: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = barrier.shape
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(barrier.copy(), m, (x, y), 2, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
    return m[1:-1, 1:-1] > 0


def fill_room(barrier: np.ndarray, seed: tuple[float, float], max_px: float, px_per_m: float,
              hint_m2: float | None = None) -> np.ndarray | None:
    """Заливка свободного пола от точки подписи.
    - узкие «утечки» (полосы штриховки, щели) уже 0.3 м отрезаются открытием, остаётся часть с точкой старта;
    - старт пробуем в нескольких свободных точках рядом с подписью (сама подпись тёмная, рядом бывает штриховка);
      если известна подписанная площадь — берём заливку, ближайшую к ней, иначе самую большую замкнутую.
    None — все варианты утекли за край листа или больше max_px."""
    h, w = barrier.shape
    x0, y0 = int(round(seed[0])), int(round(seed[1]))
    if not (0 <= x0 < w and 0 <= y0 < h):
        return None
    r = max(20, int(0.6 * px_per_m))
    ys, xs = np.nonzero(barrier[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1] == 0)
    if not len(xs):
        return None
    xs, ys = xs + max(0, x0 - r), ys + max(0, y0 - r)
    order = np.argsort((xs - x0) ** 2 + (ys - y0) ** 2)
    k = max(3, int(0.15 * px_per_m)) | 1
    cands, seen = [], np.zeros((h, w), bool)
    for i in order[::max(1, len(order) // 40)]:  # ~40 стартов от ближних к дальним
        x, y = int(xs[i]), int(ys[i])
        if seen[y, x]:
            continue
        reg = _flood(barrier, x, y)
        seen |= reg
        n = reg.sum()
        yy, xx = np.nonzero(reg)
        if n > max_px or xx.min() == 0 or yy.min() == 0 or xx.max() >= w - 1 or yy.max() >= h - 1:
            continue
        # отрезать узкие перешейки и взять часть, ближайшую к подписи
        op = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        nl, lab = cv2.connectedComponents(op, connectivity=4)
        if nl < 2:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        core = (lab == (1 + int(np.argmax(sizes)))).astype(np.uint8)
        # восстановить края комнаты, срезанные открытием: ядро + k px, но только внутри исходной заливки
        part = (cv2.dilate(core, np.ones((k + 2, k + 2), np.uint8)) > 0) & reg
        cands.append(part)
    if not cands:
        return None
    if hint_m2:
        best = min(cands, key=lambda c: abs(c.sum() / px_per_m ** 2 - hint_m2))
    else:
        best = max(cands, key=lambda c: c.sum())
    # дыры (мебель, текст, сантехника) — тоже пол комнаты
    cnts, _ = cv2.findContours(best.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    # барьер был утолщён на 1 px — возвращаем полосу у стен
    return cv2.dilate(filled, np.ones((3, 3), np.uint8)) > 0


def spaces_from_labels(img: np.ndarray, labels: list[dict], px_per_m: float, doors: list | None = None) -> list[dict]:
    """labels: [{"text", "x", "y", "value_m2"?}]. doors: [[x0,y0,x1,y1], ...] — боксы дверей (для плинтуса)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    barrier = barrier_mask(gray)
    max_px = ROOM_MAX_M2 * px_per_m ** 2
    closed = {}  # барьеры с закрытыми проёмами, строятся лениво: только если какая-то заливка утекла
    out, taken = [], np.zeros(gray.shape, bool)
    for lb in labels:
        reg = fill_room(barrier, (lb["x"], lb["y"]), max_px, px_per_m, lb.get("value_m2") or None)
        # утекла: двойные двери (щель между дугами), проёмы без двери — закрываем проёмы лучами между торцами
        # стен и пробуем ещё раз (только для этой комнаты: у остальных лучи могут рассечь комнату).
        # Стены ищем сначала толстые (наружные), потом тоньше (перегородки бывают вдвое тоньше)
        for k in (None, 5, 3):
            if reg is not None:
                break
            if k not in closed:
                walls, thick = wall_mask(gray, k=k)
                _, gaps = close_openings(walls, thick, px_per_m)
                cb = barrier.copy()
                for a, b, _d in gaps:
                    cv2.line(cb, a, b, 1, thickness=3)
                closed[k] = cb
            reg = fill_room(closed[k], (lb["x"], lb["y"]), max_px, px_per_m, lb.get("value_m2") or None)
        if reg is None:
            out.append({"name": lb.get("text", ""), "ok": False, "reason": "контур не замкнут (проём без двери или утечка)"})
            continue
        if (reg & taken).sum() > 0.5 * reg.sum():  # две подписи в одной комнате — строка остаётся (порядок = подписи)
            out.append({"name": lb.get("text", ""), "ok": False, "reason": "та же область, что у предыдущей подписи"})
            continue
        taken |= reg
        cnts, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        area = float(reg.sum())
        perim = cv2.arcLength(cv2.approxPolyDP(c, 2.0, True), True)
        # плинтус: периметр минус ширина дверей, касающихся комнаты (ширина = большая сторона бокса двери)
        ring = cv2.dilate(reg.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        door_w = []
        for b in doors or []:
            x0, y0, x1, y1 = [int(v) for v in b]
            if ring[max(0, y0):y1 + 1, max(0, x0):x1 + 1].any():
                door_w.append(max(x1 - x0, y1 - y0))
        out.append({"name": lb.get("text", ""), "ok": True, "label_m2": lb.get("value_m2") or None,
                    "polygon": cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2).tolist(),
                    "area_m2": round(area / px_per_m ** 2, 2), "perimeter_m": round(perim / px_per_m, 2),
                    "doors": len(door_w), "door_width_m": round(sum(door_w) / px_per_m, 2),
                    "baseboard_m": round(max(0.0, perim - sum(door_w)) / px_per_m, 2)})
    return out
