"""Парсер CubiCasa5K model.svg -> список (класс, полигон) в пикселях картинки.

Структура SVG (по репозиторию CubiCasa5k, floortrans/loaders/house.py):
  <g id="Wall" class="Wall ..."><polygon points="..."/> <g id="Window" ...><polygon/></g> ...</g>
  <g id="Door" class="Door ..."><polygon points="..."/></g>
  <g class="Space Kitchen"><polygon points="..."/></g>
  <g class="FixedFurniture Toilet"><g class="BoundaryPolygon"><polygon/></g></g>

Мы не доверяем этому описанию слепо: scripts/inspect_cubicasa.py считает реальные
имена классов в датасете, а неизвестные имена собираются в `unknown`.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from lxml import etree

import numpy as np

from .classes import DEFAULT_ROOM, FIXTURE_MAP, FIXTURE_SKIP, SPACE_MAP, SPACE_SKIP

Polygon = list[tuple[float, float]]


@dataclass
class ParsedSvg:
    objects: list[tuple[str, Polygon]] = field(default_factory=list)
    unknown: Counter = field(default_factory=Counter)  # имена, которые не смаппились
    width: float | None = None   # ВНИМАНИЕ: в CubiCasa это габарит содержимого, а не размер картинки.
    height: float | None = None  # Координаты полигонов = пиксели F1_scaled.png.


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", re.I)


def parse_points(s: str) -> Polygon:
    nums = [float(x) for x in _NUM.findall(s or "")]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _local(tag) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


_TF = re.compile(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)")


def parse_transform(s: str | None) -> np.ndarray:
    """SVG transform -> 3x3 матрица."""
    M = np.eye(3)
    for kind, args in _TF.findall(s or ""):
        v = [float(x) for x in _NUM.findall(args)]
        T = np.eye(3)
        if kind == "matrix" and len(v) == 6:
            T[0, :] = v[0], v[2], v[4]
            T[1, :] = v[1], v[3], v[5]
        elif kind == "translate" and v:
            T[0, 2], T[1, 2] = v[0], v[1] if len(v) > 1 else 0.0
        elif kind == "scale" and v:
            T[0, 0], T[1, 1] = v[0], v[1] if len(v) > 1 else v[0]
        elif kind == "rotate" and v:
            a = np.radians(v[0])
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            if len(v) == 3:
                R = np.array([[1, 0, v[1]], [0, 1, v[2]], [0, 0, 1]]) @ R @ np.array([[1, 0, -v[1]], [0, 1, -v[2]], [0, 0, 1]])
            T = R
        M = M @ T
    return M


def ctm(el) -> np.ndarray:
    """Полная матрица преобразования элемента (от корня до него включительно)."""
    M = np.eye(3)
    for a in reversed([el, *el.iterancestors()]):
        if a.get("transform"):
            M = M @ parse_transform(a.get("transform"))
    return M


def apply(M: np.ndarray, poly: Polygon) -> Polygon:
    if np.allclose(M, np.eye(3)):
        return poly
    P = np.c_[np.asarray(poly, float), np.ones(len(poly))] @ M.T
    return [(float(x), float(y)) for x, y in P[:, :2]]


def _first_polygon(el) -> Polygon | None:
    """Первый дочерний <polygon> (не рекурсивно в чужие объекты)."""
    for ch in el:
        if _local(ch.tag) == "polygon":
            pts = parse_points(ch.get("points", ""))
            if len(pts) >= 3:
                return pts
    return None


def _boundary_polygon(el) -> Polygon | None:
    """Контур иконки в ЛОКАЛЬНЫХ координатах (transform применяется снаружи).
    BoundaryPolygon/<polygon>; если там rect/path — габарит из <desc> (Width/Depth) или из rect-ов."""
    bp = next((g for g in el.iter() if _local(g.tag) == "g" and "BoundaryPolygon" in (g.get("class") or "")), None)
    p = _first_polygon(bp) if bp is not None else None
    if p:
        return p
    p = _first_polygon(el)
    if p:
        return p
    desc = next((d.text for d in el if _local(d.tag) == "desc" and d.text), "")
    m = dict(re.findall(r"(\w+):\s*(-?[\d.]+)", desc))
    if "Width" in m and "Depth" in m:
        w, d = float(m["Width"]), float(m["Depth"])
        return [(0, 0), (w, 0), (w, d), (0, d)]
    xs, ys = [], []
    for r in (bp if bp is not None else el).iter():
        if _local(r.tag) == "rect":
            x, y = float(r.get("x", 0)), float(r.get("y", 0))
            xs += [x, x + float(r.get("width", 0))]
            ys += [y, y + float(r.get("height", 0))]
    if xs and max(xs) > min(xs) and max(ys) > min(ys):
        return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
    return None


def _float(v):
    try:
        return float(re.sub(r"[^\d.]", "", v)) if v else None
    except ValueError:
        return None


def parse_cubicasa_svg(path_or_bytes) -> ParsedSvg:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        root = etree.fromstring(path_or_bytes)
    else:
        root = etree.parse(str(path_or_bytes)).getroot()

    out = ParsedSvg(width=_float(root.get("width")), height=_float(root.get("height")))
    if (out.width is None or out.height is None) and root.get("viewBox"):
        vb = [float(x) for x in root.get("viewBox").split()]
        out.width, out.height = vb[2], vb[3]

    for el in root.iter():
        if _local(el.tag) != "g":
            continue
        gid = el.get("id") or ""
        cls = (el.get("class") or "").strip()
        first = cls.split(" ")[0] if cls else ""

        if gid == "Wall" or first == "Wall":
            p = _first_polygon(el)
            if p:
                out.objects.append(("wall", apply(ctm(el), p)))
        elif gid == "Window" or first == "Window":
            p = _first_polygon(el)
            if p:
                out.objects.append(("window", apply(ctm(el), p)))
        elif gid == "Door" or first == "Door":
            p = _first_polygon(el)
            if p:
                out.objects.append(("door", apply(ctm(el), p)))
        elif first == "Space":
            name = cls.split(" ")[1] if " " in cls else "Undefined"
            if name in SPACE_SKIP:
                continue
            p = _first_polygon(el)
            if not p:
                continue
            target = SPACE_MAP.get(name)
            if target is None:
                out.unknown[f"Space {name}"] += 1
                target = DEFAULT_ROOM
            out.objects.append((target, apply(ctm(el), p)))
        elif first == "FixedFurniture":
            name = cls.split(" ")[1] if " " in cls else ""
            if name in FIXTURE_SKIP:
                continue
            target = FIXTURE_MAP.get(name)
            if target is None:
                out.unknown[f"FixedFurniture {name}"] += 1
                continue
            p = _boundary_polygon(el)
            if p:
                out.objects.append((target, apply(ctm(el), p)))
            else:
                out.unknown[f"no-shape {name}"] += 1
    return out
