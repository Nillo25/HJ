"""«Граф сцены»: объекты с id/class/polygon/conf + комнаты, сериализация в JSON и SVG."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from shapely.geometry import Point, Polygon as ShPolygon
from shapely.validation import make_valid

from .classes import COLORS, ROOM_CLASSES

PREFIX = {"wall": "w", "door": "d", "window": "win", "toilet": "t", "sink": "s",
          "bathtub": "bt", "shower": "sh"}


@dataclass
class Element:
    id: str
    cls: str
    polygon: list[list[float]]
    conf: float
    source: str = "yolo"          # yolo | gt | ocr | vlm ...
    primary: bool = True          # False = второй независимый источник для сверки
    room: str | None = None

    def shape(self) -> ShPolygon:
        p = ShPolygon(self.polygon)
        return p if p.is_valid else make_valid(p).buffer(0)


@dataclass
class Scene:
    image: str
    width: int
    height: int
    objects: list[Element] = field(default_factory=list)
    rooms: list[Element] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---------- построение ----------
    @classmethod
    def build(cls, image: str, width: int, height: int,
              detections: list[tuple[str, list, float]], source: str = "yolo", meta=None) -> "Scene":
        sc = cls(image=image, width=width, height=height, meta=meta or {})
        counters: Counter = Counter()
        for c, poly, conf in detections:
            if len(poly) < 3:
                continue
            counters[c] += 1
            pts = [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
            if c in ROOM_CLASSES:
                rid = f"{c.removeprefix('room_')}_{counters[c]}"
                sc.rooms.append(Element(rid, c, pts, float(conf), source))
            else:
                oid = f"{PREFIX.get(c, c)}{counters[c]}"
                sc.objects.append(Element(oid, c, pts, float(conf), source))
        sc.assign_rooms()
        return sc

    def assign_rooms(self):
        """Привязка объекта к комнате: центр объекта внутри полигона комнаты,
        иначе комната с наибольшим пересечением, иначе ближайшая (двери/окна в стенах)."""
        room_shapes = [(r.id, r.shape()) for r in self.rooms]
        for o in self.objects:
            if o.cls == "wall" or not room_shapes:
                continue
            s = o.shape()
            c = s.representative_point() if not s.is_empty else Point(o.polygon[0])
            hit = [rid for rid, rs in room_shapes if rs.contains(c)]
            if hit:
                o.room = hit[0]
                continue
            best = max(room_shapes, key=lambda t: t[1].intersection(s).area)
            if best[1].intersection(s).area > 0:
                o.room = best[0]
                continue
            near = min(room_shapes, key=lambda t: t[1].distance(c))
            if near[1].distance(c) < 25:  # px
                o.room = near[0]

    # ---------- сериализация ----------
    def to_dict(self) -> dict:
        def el(e: Element):
            d = asdict(e)
            d["class"] = d.pop("cls")
            return d
        return {"image": self.image, "width": self.width, "height": self.height, "meta": self.meta,
                "objects": [el(o) for o in self.objects],
                "rooms": [{**el(r), "type": r.cls.removeprefix("room_")} for r in self.rooms]}

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        def el(x):
            return Element(x["id"], x["class"], x["polygon"], x["conf"], x.get("source", "yolo"),
                           x.get("primary", True), x.get("room"))
        return cls(d["image"], d["width"], d["height"], [el(o) for o in d["objects"]],
                   [el(r) for r in d["rooms"]], d.get("meta", {}))

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "scene.json").write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
        (folder / "scene.svg").write_text(self.to_svg(image_href=self.image))

    @classmethod
    def load(cls, folder: Path) -> "Scene":
        return cls.from_dict(json.loads((folder / "scene.json").read_text()))

    def to_svg(self, image_href: str | None = None) -> str:
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
               f'viewBox="0 0 {self.width} {self.height}" width="{self.width}" height="{self.height}">']
        css = " ".join(f".{c}{{--c:{col}}}" for c, col in COLORS.items())
        out.append(f"<style>{css} polygon{{fill:var(--c);stroke:var(--c);fill-opacity:.25;stroke-width:2}}"
                   f" .rooms polygon{{fill-opacity:.15;stroke-dasharray:6 4}}</style>")
        if image_href:
            out.append(f'<image href="{escape(image_href)}" x="0" y="0" width="{self.width}" height="{self.height}"/>')
        for group, items in (("rooms", self.rooms), ("objects", self.objects)):
            out.append(f'<g class="{group}">')
            for e in items:
                pts = " ".join(f"{x},{y}" for x, y in e.polygon)
                room = f' data-room="{e.room}"' if e.room else ""
                z = (self.meta.get("zones") or {}).get(e.id)
                style = f' style="--c:{z["color"]}" data-label="{escape(z["label"])}"' if z else ""
                out.append(f'<polygon id="{e.id}" class="{e.cls}"{room}{style} data-conf="{e.conf:.2f}" points="{pts}"/>')
            out.append("</g>")
        out.append("</svg>")
        return "\n".join(out)
