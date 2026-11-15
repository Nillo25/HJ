"""Исправления и данные от пользователя через чат: «VH — это гардеробная», «это не спальня, а кабинет»,
«эта дверь лишняя», «на плане 12 окон», «отмени».

Исправление меняет сцену плана (scene.json) и пишется в журнал meta["corrections"] — его видно и можно отменить.
Обозначения (define_term) — в общий словарь пользователя (data/glossary_user.json), действуют на все планы.
Ответ на правку считает наш код, как и остальные ответы; кэш ответов не хранит правки (зависит от сцены).
"""
from __future__ import annotations

import time
from dataclasses import asdict

from .classes import LABELS_RU, ROOM_CLASSES
from .scene import Element, Scene

ROOM_TYPES = sorted(c.removeprefix("room_") for c in ROOM_CLASSES)


def _log(scene: Scene, entry: dict) -> None:
    scene.meta.setdefault("corrections", []).append({**entry, "at": round(time.time())})
    scene.meta["_dirty"] = True


def _answer(value, note: str, ids: list | None = None, action: str | None = "highlight"):
    from .query import Answer
    return Answer(value, None, "high", 1.0, ids or [], [{"kind": "user"}], note, action=action if ids else None)


def define_term(scene: Scene, term: str, kind: str) -> "Answer":
    """Обозначение на планах -> вид помещения (общий словарь). На этом листе типы комнат по подписям — заново."""
    from . import glossary
    m = glossary.add_term(term, kind)
    changed = []
    sheet = scene.meta.get("sheet") or {}
    if scene.meta.get("room_types_from_labels", {}).get("done"):
        from .infer import room_types_from_labels
        changed = room_types_from_labels(scene, sheet).get("changed", [])
    _log(scene, {"tool": "define_term", "term": term, "kind": kind})
    note = f"запомнил: «{term}» — {glossary.kinds()[kind].get('ru', kind)} (для всех планов)"
    if changed:
        note += f"; на этом плане тип изменён у {len(changed)} помещ."
    return _answer({"term": term, "kind": kind, "matched": m.get("term")}, note, [c["id"] for c in changed])


def set_room_type(scene: Scene, room: str, type: str) -> "Answer":
    """Тип помещения по id (bedroom_1) — «это не спальня, а кабинет» -> other."""
    t = type.removeprefix("room_")
    if t not in ROOM_TYPES:
        raise ValueError(f"тип {type!r} не из списка: {', '.join(ROOM_TYPES)}")
    r = next((r for r in scene.rooms if r.id == room), None)
    if r is None:
        raise ValueError(f"помещения {room!r} нет; есть: {', '.join(x.id for x in scene.rooms)[:400]}")
    was = r.cls
    r.cls = f"room_{t}"
    r.conf, r.source = 1.0, "user"
    _log(scene, {"tool": "set_room_type", "room": room, "was": was, "now": r.cls})
    return _answer({"room": room, "was": was.removeprefix("room_"), "now": t},
                   f"{room}: {LABELS_RU.get(was, was)} -> {LABELS_RU.get(r.cls, r.cls)}", [room])


def remove_objects(scene: Scene, ids: list[str]) -> "Answer":
    """Убрать ошибочно найденные объекты или помещения по id («эта дверь лишняя»). Отменяется undo_correction."""
    ids = list(dict.fromkeys(ids))
    objs = [o for o in scene.objects if o.id in ids]
    rooms = [r for r in scene.rooms if r.id in ids]
    missing = sorted(set(ids) - {e.id for e in objs + rooms})
    if not objs and not rooms:
        raise ValueError(f"таких id на плане нет: {', '.join(missing)}")
    gone = {e.id for e in objs + rooms}
    scene.objects = [o for o in scene.objects if o.id not in gone]
    scene.rooms = [r for r in scene.rooms if r.id not in gone]
    if rooms:
        scene.assign_rooms()
    _log(scene, {"tool": "remove_objects", "objects": [asdict(o) for o in objs], "rooms": [asdict(r) for r in rooms]})
    by: dict[str, int] = {}
    for e in objs + rooms:
        by[e.cls] = by.get(e.cls, 0) + 1
    note = "убрано: " + ", ".join(f"{LABELS_RU.get(c, c)} {n}" for c, n in by.items())
    if missing:
        note += f"; не найдены: {', '.join(missing)}"
    return _answer(sorted(gone), note, action=None)


ADDABLE = {"window", "door", "toilet", "sink", "bathtub", "shower", "sauna_bench"}
QUERY_CLASS = {"bathtub": "shower_bathtub", "shower": "shower_bathtub", "sauna_bench": "sauna"}   # класс в вопросе «сколько …»


def _count_now(scene: Scene, qcls: str):
    from .query import count
    try:
        return count(scene, qcls).answer
    except Exception:  # noqa: BLE001
        return None


def add_object(scene: Scene, cls: str, x: float, y: float, polygon: list | None = None, source: str = "user") -> dict:
    """Пропущенный объект («+ Add»): контур от fp/add_at (модель или ИИ по вырезке у клика); без контура — рамка
    размером с медиану таких же объектов. Если ответ «сколько» этот объект не учитывает (число даёт опись Opus),
    число поднимается на 1 правкой count_overrides. В журнале правок, отменяется undo_correction."""
    if cls not in ADDABLE:
        raise ValueError(f"добавить можно только: {', '.join(sorted(ADDABLE))}")
    if not (0 <= x <= scene.width and 0 <= y <= scene.height):
        raise ValueError("точка вне плана")
    same = [o for o in scene.objects if o.cls == cls]
    if polygon:
        poly = [[round(float(px), 1), round(float(py), 1)] for px, py in polygon]
    else:
        if same:
            ws = sorted(max(p[0] for p in o.polygon) - min(p[0] for p in o.polygon) for o in same)
            hs = sorted(max(p[1] for p in o.polygon) - min(p[1] for p in o.polygon) for o in same)
            w, h = ws[len(ws) // 2], hs[len(hs) // 2]
        else:
            ppm = scene.meta.get("px_per_m") or 0
            w = h = ppm * 1.0 if ppm else scene.width * 0.015
        w, h = max(w, 6.0), max(h, 6.0)
        poly = [[round(x - w / 2, 1), round(y - h / 2, 1)], [round(x + w / 2, 1), round(y - h / 2, 1)],
                [round(x + w / 2, 1), round(y + h / 2, 1)], [round(x - w / 2, 1), round(y + h / 2, 1)]]
    prefix = (same[0].id.rstrip("0123456789") if same else cls[:3]) or cls
    used = {o.id for o in scene.objects}
    n = 1 + max((int(i[len(prefix):]) for i in used if i.startswith(prefix) and i[len(prefix):].isdigit()), default=0)
    oid = f"{prefix}{n}"
    qcls = QUERY_CLASS.get(cls, cls)
    before = _count_now(scene, qcls)
    scene.objects.append(Element(id=oid, cls=cls, polygon=poly, conf=1.0, source=source))
    scene.assign_rooms()
    ov = scene.meta.setdefault("count_overrides", {})
    prev = ov.get(qcls)
    after = _count_now(scene, qcls)
    bumped = isinstance(before, int) and after == before
    if bumped:
        ov[qcls] = before + 1
        after = before + 1
    _log(scene, {"tool": "add_object", "id": oid, "cls": cls, "qcls": qcls, "bumped": bumped, "prev_override": prev})
    return {"id": oid, "cls": cls, "count": after}


def mark_room(scene: Scene, x: float, y: float, kind: str) -> dict:
    """«+ Add» по комнате: клик внутри помещения + «это кухня» -> тип помещения (самое маленькое, что содержит точку).
    Если ответ «сколько кухонь» правку не учитывает (число даёт опись Opus), число поднимается на 1. Отменяется undo."""
    from shapely.geometry import Point
    t = kind.removeprefix("room_")
    if t not in ROOM_TYPES:
        raise ValueError(f"тип помещения {kind!r} не из списка: {', '.join(ROOM_TYPES)}")
    pt = Point(x, y)
    inside = [r for r in scene.rooms if r.shape().contains(pt)]
    if not inside:
        raise ValueError("здесь нет найденного помещения: кликните внутри комнаты")
    r = min(inside, key=lambda e: e.shape().area)
    now = f"room_{t}"
    if r.cls == now:
        raise ValueError(f"это помещение уже {LABELS_RU.get(now, t)}")
    before = _count_now(scene, now)
    was = r.cls
    r.cls, r.conf, r.source = now, 1.0, "user"
    ov = scene.meta.setdefault("count_overrides", {})
    prev = ov.get(now)
    after = _count_now(scene, now)
    bumped = isinstance(before, int) and after == before
    if bumped:
        ov[now] = before + 1
        after = before + 1
    _log(scene, {"tool": "mark_room", "room": r.id, "was": was, "now": now, "qcls": now, "bumped": bumped,
                 "prev_override": prev})
    return {"id": r.id, "cls": now, "count": after}


def set_count(scene: Scene, cls: str, value: int) -> "Answer":
    """Число объектов класса на всём плане со слов пользователя — дальше подсчёт отвечает им (уверенность high)."""
    from .query import _count_plan
    if value < 0:
        raise ValueError("число не может быть отрицательным")
    ov = scene.meta.setdefault("count_overrides", {})
    prev = ov.get(cls)
    try:
        before = prev if prev is not None else _count_plan(scene, cls).answer
    except Exception:  # noqa: BLE001
        before = None
    ov[cls] = int(value)
    _log(scene, {"tool": "set_count", "cls": cls, "was": before, "prev_override": prev, "now": int(value)})
    return _answer({"cls": cls, "was": before, "now": int(value)},
                   f"{LABELS_RU.get(cls, cls)}: теперь {int(value)} (было {before}) — со слов пользователя", action=None)


def undo_correction(scene: Scene) -> "Answer":
    """Отменить последнее исправление на этом плане (обозначение остаётся в словаре)."""
    log = scene.meta.get("corrections") or []
    if not log:
        return _answer(None, "на этом плане исправлений нет", action=None)
    e = log.pop()
    if e["tool"] == "set_room_type":
        r = next((r for r in scene.rooms if r.id == e["room"]), None)
        if r:
            r.cls = e["was"]
    elif e["tool"] == "remove_objects":
        scene.objects += [Element(**o) for o in e["objects"]]
        scene.rooms += [Element(**r) for r in e["rooms"]]
        scene.assign_rooms()
    elif e["tool"] == "mark_room":
        r = next((r for r in scene.rooms if r.id == e["room"]), None)
        if r:
            r.cls = e["was"]
        if e.get("bumped"):
            ov = scene.meta.setdefault("count_overrides", {})
            if e.get("prev_override") is None:
                ov.pop(e["qcls"], None)
            else:
                ov[e["qcls"]] = e["prev_override"]
    elif e["tool"] == "add_object":
        scene.objects = [o for o in scene.objects if o.id != e["id"]]
        if e.get("bumped"):
            ov = scene.meta.setdefault("count_overrides", {})
            if e.get("prev_override") is None:
                ov.pop(e["qcls"], None)
            else:
                ov[e["qcls"]] = e["prev_override"]
    elif e["tool"] == "set_count":
        ov = scene.meta.setdefault("count_overrides", {})
        if e.get("prev_override") is None:
            ov.pop(e["cls"], None)
        else:
            ov[e["cls"]] = e["prev_override"]
    scene.meta["_dirty"] = True
    return _answer(e["tool"], f"отменено: {e['tool']}" + (" (обозначение в словаре осталось)" if e["tool"] == "define_term" else ""),
                   action=None)


def list_corrections(scene: Scene) -> "Answer":
    """Журнал исправлений этого плана."""
    log = [{k: v for k, v in e.items() if k not in ("objects", "rooms")} |
           ({"removed": [o["id"] for o in e.get("objects", []) + e.get("rooms", [])]} if e["tool"] == "remove_objects" else {})
           for e in scene.meta.get("corrections") or []]
    return _answer(log, f"исправлений: {len(log)}", action=None)


FUNCS = {"define_term": define_term, "set_room_type": set_room_type, "remove_objects": remove_objects,
         "set_count": set_count, "undo_correction": undo_correction, "list_corrections": list_corrections}
