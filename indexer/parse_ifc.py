"""IFC -> elements, через ifcopenshell (тот же инструмент, которым Stoa
считали правильные ответы — это видно в подписи их PDF-планов).

Проверено на превью:
  стены   = IfcWall + IfcWallStandardCase   (99 + 747 = 846 = их ответ)
  сантех. = IfcFlowTerminal                  (119 = их ответ; IfcSanitaryTerminal даёт 0)
  по этажам: 42/54 через ContainedInSpatialStructure, остальные 12 — помещения,
  они привязаны через IfcRelAggregates. Здесь учтены обе связи.
"""
import json

import ifcopenshell
import ifcopenshell.util.element as ue

# тип вопроса -> классы IFC
TYPES = {
    "door":      ["IfcDoor"],
    "window":    ["IfcWindow"],
    "room":      ["IfcSpace"],
    "wall":      ["IfcWall", "IfcWallStandardCase"],
    "column":    ["IfcColumn"],
    "beam":      ["IfcBeam"],
    "slab":      ["IfcSlab"],
    "stair":     ["IfcStair", "IfcStairFlight"],
    "railing":   ["IfcRailing"],
    "fixture":   ["IfcFlowTerminal", "IfcSanitaryTerminal"],
    "furniture": ["IfcFurnishingElement"],
    "covering":  ["IfcCovering"],
}

# в каком поле Qto лежит площадь помещения — зависит от программы-автора
AREA_KEYS = ("NetFloorArea", "GrossFloorArea", "NetArea", "GrossArea", "Area")
VOL_KEYS = ("NetVolume", "GrossVolume", "Volume")
LEN_KEYS = ("Length", "NetLength")


def _qto(el, keys):
    """Ищем количество в наборах Qto_*, имена у всех программ разные."""
    try:
        psets = ue.get_psets(el, qtos_only=True)
    except Exception:
        return None
    for _, props in (psets or {}).items():
        for k in keys:
            v = props.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _storey_of(el):
    """Этаж элемента. Двери и стены — через ContainedInSpatialStructure,
    помещения — через Aggregates (декомпозицию). Проверяем обе."""
    try:
        for rel in getattr(el, "ContainedInStructure", []) or []:
            s = rel.RelatingStructure
            if s and s.is_a("IfcBuildingStorey"):
                return s
    except Exception:
        pass
    try:
        p = ue.get_container(el)          # поднимается по Aggregates
        while p is not None:
            if p.is_a("IfcBuildingStorey"):
                return p
            p = ue.get_container(p)
    except Exception:
        pass
    try:
        for rel in getattr(el, "Decomposes", []) or []:
            s = rel.RelatingObject
            if s and s.is_a("IfcBuildingStorey"):
                return s
    except Exception:
        pass
    return None


def parse(path: str, dict_rows: dict):
    """-> (floors, elements). floors: [{name, level, ord}];
    elements несут floor_name, индексатор свяжет с floors."""
    f = ifcopenshell.open(path)

    storeys = []
    for s in f.by_type("IfcBuildingStorey"):
        storeys.append({
            "name": getattr(s, "Name", None) or getattr(s, "LongName", None) or "",
            "level": float(getattr(s, "Elevation", 0) or 0),
            "guid": s.GlobalId,
        })
    storeys.sort(key=lambda x: x["level"])
    for i, s in enumerate(storeys):
        s["ord"] = i

    from labels import norm, subtype_for

    out = []
    seen = set()
    for t, classes in TYPES.items():
        for cls in classes:
            try:
                items = f.by_type(cls)
            except Exception:
                continue
            for el in items:
                if el.id() in seen:        # IfcWallStandardCase наследует IfcWall
                    continue
                seen.add(el.id())
                st = _storey_of(el)

                # у помещения два имени: Name — номер («4-3»),
                # LongName — название («TUALETTRUUM»). В вопросах они склеены.
                name = getattr(el, "Name", None) or ""
                longname = getattr(el, "LongName", None) or ""
                label = (f"{longname} ({name})".strip()
                         if t == "room" and longname and name
                         else (longname or name or None))

                row = {
                    "type": t,
                    "subtype": subtype_for(longname or name, dict_rows) if t == "room" else None,
                    "label": label,
                    "label_norm": norm(longname or name) if label else None,
                    "source_ref": el.GlobalId,
                    "floor_name": st.Name if st is not None else None,
                    "props": json.dumps({"ifc_class": el.is_a(),
                                         "name": name, "long_name": longname}),
                }
                if t == "room":
                    row["area"] = _qto(el, AREA_KEYS)
                    row["volume"] = _qto(el, VOL_KEYS)
                elif t in ("wall", "beam"):
                    row["length"] = _qto(el, LEN_KEYS)
                elif t == "slab":
                    row["volume"] = _qto(el, VOL_KEYS)
                    row["area"] = _qto(el, AREA_KEYS)
                out.append(row)

    return storeys, out
