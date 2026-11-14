"""Стены по границам помещений на чертёжных листах, где модель стен не видит.

На механических/конструктивных листах (M-101A) архитектура — бледный фон из пар тонких линий; модель,
обученная на квартирах (стена = толстая чёрная полоса), находит единицы участков. Искать стены по линиям
чертежа ненадёжно: трубы нарисованы так же (двойные серые линии). Помещения же на таких листах уже есть
(рамки ИИ, подогнанные к линиям стен: fp/rooms_cv, fp.infer.rooms_by_walls) — их границы и есть стены.

Добавляется только то, чего нет: граница помещения, не покрытая найденными стенами (доля покрытия ниже
MIN_COVER — значит модель стен тут не видит), становится полосой толщиной WALL_M; общие стены соседних
помещений склеиваются. На листах, где модель стены видит (школа: ~270 участков), ничего не меняется.
"""
from __future__ import annotations

from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

WALL_M = 0.2          # толщина рисуемой стены
MIN_COVER = 0.5       # доля периметра помещений, покрытая стенами модели; выше — стены уже есть, не трогаем
MIN_PIECE_M = 0.5     # короче — обрывки на углах
# только инженерные листы (M — механика, P — сантехника, E — электрика, F — пожарная): там архитектура — бледная
# подложка, модель её не видит. На конструктивных (S) «помещения» по стенам — панели плиты между швами (S-100F:
# стены легли бы на деформационные швы), на архитектурных модель стены видит сама
MEP_PREFIXES = ("M", "P", "E", "F")


def applies(sheet_number: str | None) -> bool:
    return bool(sheet_number) and str(sheet_number).strip().upper().startswith(MEP_PREFIXES)


def _edge_bands(rooms, half: float) -> list:
    """Полосы вдоль каждой стороны каждого помещения (без колец с дырами — элемент сцены хранит один контур)."""
    bands = []
    for r in rooms:
        try:
            pts = list(r.shape().exterior.coords)
        except AttributeError:
            continue
        for a, b in zip(pts, pts[1:]):
            seg = LineString([a, b])
            if seg.length > 0:
                bands.append(seg.buffer(half, cap_style=2, join_style=2))
    return bands


def add(sc, px_per_m: float) -> dict:
    """Добавить в сцену стены по границам помещений там, где стен модели нет."""
    from .scene import Element
    # только помещения, построенные по линиям стен чертежа (fp.infer.rooms_by_walls): у контуров модели граница
    # гуляет относительно оси стены, и там, где модель видит стены (школа), стены по ним лишние
    rooms = [r for r in sc.rooms if r.cls != "room_outdoor" and r.source == "vlm+walls"]
    if not rooms or not px_per_m:
        return {"done": False, "why": "нет помещений или масштаба"}
    half = WALL_M * px_per_m / 2
    walls = [o.shape() for o in sc.objects if o.cls == "wall"]
    wall_union = unary_union(walls).buffer(half) if walls else None
    perim = unary_union([r.shape().exterior for r in rooms])
    covered = perim.intersection(wall_union).length / perim.length if wall_union is not None and perim.length else 0.0
    if covered >= MIN_COVER:
        return {"done": False, "cover": round(covered, 2), "why": "стены модели покрывают границы помещений"}
    taken = wall_union
    n = 0
    for band in _edge_bands(rooms, half):
        piece = band.difference(taken) if taken is not None else band  # общие стены соседних комнат — один раз
        geoms = list(piece.geoms) if isinstance(piece, MultiPolygon) else [piece]
        for gm in geoms:
            if not isinstance(gm, Polygon) or gm.is_empty:
                continue
            x0, y0, x1, y1 = gm.bounds
            if max(x1 - x0, y1 - y0) < MIN_PIECE_M * px_per_m:
                continue
            n += 1
            pts = [[round(x, 1), round(y, 1)] for x, y in gm.exterior.coords[:-1]]
            sc.objects.append(Element(f"wr{n}", "wall", pts, 0.7, source="rooms"))
        taken = band if taken is None else taken.union(band)
    return {"done": True, "cover_before": round(covered, 2), "added": n, "rooms": len(rooms)}
