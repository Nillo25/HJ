"""Функции-инструменты над графом сцены + расчёт confidence по правилам из confidence.md.

LLM в пиксели не смотрит — вызывает только эти функции.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .classes import CLASSES, LABELS_RU, ROOM_CLASSES
from .scene import Element, Scene

import json as _json
from pathlib import Path as _Path

CAL = _json.loads((_Path(__file__).parent / "thresholds.json").read_text())


def _calibrated(scene: Scene, items: list[Element], classes: list[str], room_ids: set | None):
    """Уровень уверенности подсчёта по калибровке (scripts/tune_thresholds.py calibrate):
    near = пограничные детекции (принятые с conf < порог+margin и отброшенные с conf >= порог-margin).
    None — калибровки нет (разметка gt или старая сцена без candidates)"""
    if "candidates" not in scene.meta or len(classes) != 1 or classes[0] not in CAL:
        return None
    cls, cal = classes[0], CAL[classes[0]]
    if "medium_max_near" not in cal:
        return None
    m = CAL.get("_calibration", {}).get("near_margin", 0.15)
    near = sum(1 for e in items if e.conf < cal["conf"] + m)
    for c, conf, x, y in scene.meta["candidates"]:
        if c != cls:
            continue
        if room_ids is not None:
            from shapely.geometry import Point
            if not any(r.shape().contains(Point(x, y)) for r in scene.rooms if r.id in room_ids):
                continue
        near += 1
    minconf = min((e.conf for e in items), default=1.0)
    hi = cal.get("high_minconf")
    lvl = "high" if hi is not None and near == 0 and minconf >= hi else \
        "medium" if near <= cal["medium_max_near"] else "low"
    note = f"пограничных детекций: {near} — проверьте" if lvl == "low" else None
    return lvl, round(minconf, 2), note

HIGH, MEDIUM = 0.9, 0.7       # стартовые пороги, подстраиваются по стенду оценки
ZERO_CAP, DISAGREE_CAP, AGREE_FLOOR = 0.4, 0.4, 0.95


def level(score: float) -> str:
    return "high" if score >= HIGH else "medium" if score >= MEDIUM else "low"


@dataclass
class Answer:
    answer: object
    unit: str | None = None
    confidence: str = "high"
    score: float = 1.0
    ids: list[str] = field(default_factory=list)       # что подсветить на плане
    evidence: list[dict] = field(default_factory=list)
    note: str | None = None
    action: str | None = None                          # highlight | clear | None

    def to_dict(self):
        return asdict(self)


def _ev(e: Element) -> dict:
    return {"kind": e.source, "id": e.id, "class": e.cls, "conf": round(e.conf, 2), "room": e.room}


def _resolve_classes(cls: str) -> list[str]:
    if cls == "room":
        return sorted(ROOM_CLASSES)
    if cls == "sauna":
        return ["sauna_bench"]
    if cls not in CLASSES:
        raise ValueError(f"неизвестный класс {cls!r}; доступны: {', '.join(CLASSES + ['room'])}")
    return [cls]


def _all(scene: Scene) -> list[Element]:
    return scene.objects + scene.rooms


def _find_room(scene: Scene, room: str) -> list[Element]:
    """Комната по id ('bath_1') или по типу ('bath' / 'room_bath')."""
    exact = [r for r in scene.rooms if r.id == room]
    if exact:
        return exact
    t = room if room.startswith("room_") else f"room_{room}"
    return [r for r in scene.rooms if r.cls == t]


def _score(items: list[Element], others: list[Element]) -> tuple[float, str | None]:
    """Шаг 2 из confidence.md: min по элементам, ноль подозрителен, сверка со вторым источником."""
    score = min((e.conf for e in items), default=0.0)
    note = None
    if not items:
        score = min(score, ZERO_CAP)
        note = "ничего не найдено — возможно, распознавание не справилось, проверьте"
    if others:
        if len(others) == len(items):
            score = max(score, AGREE_FLOOR)
        else:
            score = min(score, DISAGREE_CAP)
            note = f"источники разошлись: {len(items)} против {len(others)}"
    if items and score < MEDIUM and note is None:
        weak = [e.id for e in items if e.conf < MEDIUM]
        note = f"низкая уверенность у {', '.join(weak[:10])} — проверьте"
    return score, note


def _select(scene: Scene, cls: str, room: str | None, primary: bool):
    classes = _resolve_classes(cls)
    items = [e for e in _all(scene) if e.cls in classes and e.primary == primary]
    if room:
        rooms = _find_room(scene, room)
        ids = {r.id for r in rooms}
        items = [e for e in items if e.room in ids]
    return items


def count(scene: Scene, cls: str, room: str | None = None) -> Answer:
    if room and not _find_room(scene, room):
        return Answer(0, confidence="low", score=0.3, note=f"комната {room!r} не найдена")
    items = _select(scene, cls, room, True)
    others = _select(scene, cls, room, False)
    score, note = _score(items, others)
    cal = None if others else _calibrated(scene, items, _resolve_classes(cls),
                                          {r.id for r in _find_room(scene, room)} if room else None)
    if cal:  # калиброванный уровень вместо порогов HIGH/MEDIUM по min(conf)
        lvl, score, note = cal
        n = _count_saunas(items) if cls == "sauna" else len(items)
        return Answer(n, "шт", lvl, score, [e.id for e in items], [_ev(e) for e in items],
                      note or ("сауна = помещение со скамьями" if cls == "sauna" and n else None))
    if cls == "sauna":  # сауна = помещение со скамьями (в CubiCasa сауну рисуют полками, часто внутри Bath)
        n = _count_saunas(items)
        return Answer(n, "шт", level(score), round(score, 2), [e.id for e in items],
                      [_ev(e) for e in items], note or ("сауна = помещение со скамьями" if n else None))
    return Answer(len(items), "шт", level(score), round(score, 2), [e.id for e in items],
                  [_ev(e) for e in items], note)


def _count_saunas(benches: list[Element], gap: float = 40.0) -> int:
    """Сауна = комната со скамьями; скамьи без комнаты, лежащие ближе gap px друг к другу, — одна сауна."""
    rooms = {e.room for e in benches if e.room}
    loose = [e.shape() for e in benches if not e.room]
    parent = list(range(len(loose)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(loose)):
        for j in range(i + 1, len(loose)):
            if loose[i].distance(loose[j]) <= gap:
                parent[find(i)] = find(j)
    return len(rooms) + len({find(i) for i in range(len(loose))})


def highlight(scene: Scene, cls: str, room: str | None = None) -> Answer:
    a = count(scene, cls, room)
    a.action = "highlight"
    return a


def highlight_ids(scene: Scene, ids: list[str]) -> Answer:
    by_id = {e.id: e for e in _all(scene)}
    items = [by_id[i] for i in ids if i in by_id]
    score, note = _score(items, [])
    return Answer(len(items), "шт", level(score), round(score, 2), [e.id for e in items],
                  [_ev(e) for e in items], note, action="highlight")


def list_rooms(scene: Scene) -> Answer:
    rooms = [{"id": r.id, "type": r.cls.removeprefix("room_"), "name_ru": LABELS_RU.get(r.cls),
              "conf": round(r.conf, 2), "objects": sum(o.room == r.id for o in scene.objects)}
             for r in scene.rooms]
    score, note = _score(scene.rooms, [])
    return Answer(rooms, None, level(score), round(score, 2), [r.id for r in scene.rooms],
                  [_ev(r) for r in scene.rooms], note)


def objects_in(scene: Scene, room: str) -> Answer:
    rooms = _find_room(scene, room)
    if not rooms:
        return Answer([], confidence="low", score=0.3, note=f"комната {room!r} не найдена")
    ids = {r.id for r in rooms}
    items = [o for o in scene.objects if o.room in ids and o.cls != "wall"]
    score, note = _score(items + rooms, [])
    summary: dict[str, int] = {}
    for o in items:
        summary[o.cls] = summary.get(o.cls, 0) + 1
    return Answer(summary, "шт", level(score), round(score, 2), [o.id for o in items] + list(ids),
                  [_ev(o) for o in items], note, action="highlight")


def area(scene: Scene, room: str, px_per_m: float | None = None) -> Answer:
    rooms = _find_room(scene, room)
    if not rooms:
        return Answer(None, confidence="low", score=0.3, note=f"комната {room!r} не найдена")
    px_per_m = px_per_m or scene.meta.get("px_per_m")
    scale = scene.meta.get("scale") or {}
    labels = {lb["room"]: lb for lb in scale.get("labels", [])}  # подписи площадей, прочитанные с плана
    res, agree, disagree = [], 0, []
    for r in rooms:
        a = r.shape().area
        item = {"id": r.id, "area_px2": round(a), "area_m2": round(a / px_per_m ** 2, 2) if px_per_m else None}
        lb = labels.get(r.id)
        if lb:
            item["label_m2"] = lb["label_m2"]  # второй, независимый источник — подпись на чертеже
            if item["area_m2"]:
                if abs(item["area_m2"] / lb["label_m2"] - 1) <= 0.10:
                    agree += 1
                else:
                    disagree.append(f"{r.id}: контур {item['area_m2']} м², подпись {lb['label_m2']} м²")
        res.append(item)
    score, note = _score(rooms, [])
    # найдено по типу, а не по точному id, и таких комнат несколько — неоднозначно
    if len(rooms) > 1:
        score = min(score, 0.8)
        note = f"комнат такого типа {len(rooms)} — указаны все"
    if not px_per_m:
        note = (note + "; " if note else "") + "масштаб плана неизвестен — площадь в пикселях²"
    else:
        # уверенность площади не выше уверенности масштаба; контур и подпись совпали (±10%) — high
        cap = {"high": 1.0, "medium": 0.8, "low": 0.5}.get(scale.get("confidence"), 0.8)
        score = min(score, cap)
        if agree and not disagree and len(rooms) == 1:
            score = max(score, AGREE_FLOOR)
        if disagree:
            score = min(score, DISAGREE_CAP)
            note = (note + "; " if note else "") + "контур и подпись расходятся (" + "; ".join(disagree) + ")"
    unit = "m2" if px_per_m else "px2"
    return Answer(res, unit, level(score), round(score, 2), [r.id for r in rooms], [_ev(r) for r in rooms],
                  note, action="highlight")


def clear(scene: Scene) -> Answer:
    return Answer(None, action="clear")


SQFT_M2 = 0.09290304
_STOP = {"the", "and", "of", "area", "floor", "total", "1st", "2nd", "3rd", "first", "second", "third"}


def _words(s: str) -> set[str]:
    import re
    return {w for w in re.findall(r"[a-zа-яё0-9]+", s.lower()) if len(w) > 2 and w not in _STOP}


def legend(scene: Scene, label: str | None = None) -> Answer:
    """Данные листа, прочитанные один раз при загрузке: штамп, масштаб, легенда, таблицы, примечания.
    С label — зоны этого цвета легенды на плане: подсветка, площадь по масштабу, сверка с таблицей площадей."""
    sheet = scene.meta.get("sheet")
    if not sheet:
        return Answer(None, confidence="low", score=0.3,
                      note="лист не прочитан (нет ключа ИИ при загрузке) — загрузите план заново или нажмите «Авто-масштаб»")
    zones = scene.meta.get("zones") or {}
    if not label:
        summary = {"title_block": sheet.get("title_block"), "scale_text": sheet.get("scale_text"),
                   "north": sheet.get("north"),
                   "legend": [{"label": lg["label"], "zones_on_plan": sum(z["label"] == lg["label"] for z in zones.values())}
                              for lg in sheet.get("legend", [])],
                   "tables": sheet.get("tables", []), "notes": sheet.get("notes", []),
                   "dimensions": sheet.get("dimensions", [])}
        # прочитано VLM и не сверено с другим источником — не выше medium
        return Answer(summary, None, "medium", 0.8, note="прочитано с листа ИИ — сверяйте важные числа с чертежом")
    want = label.lower().strip()
    labels = [lg["label"] for lg in sheet.get("legend", [])]

    def sim(l: str) -> float:  # «accessible dormitory» -> ACCESSIBLE (ABA) DORMITORY, «dormitory» -> DORMITORY
        if l.lower() == want:
            return 2.0
        a, b = _words(l), _words(want)
        return len(a & b) / len(a | b) if a | b else 0.0
    best = max(labels, key=sim, default=None)
    if not best or sim(best) == 0:
        return Answer(None, confidence="low", score=0.3,
                      note=f"в легенде нет {label!r}; есть: {', '.join(labels) or 'ничего'}")
    ids = [zid for zid, z in zones.items() if z["label"] == best]
    items = [e for e in scene.objects if e.id in ids]
    px = scene.meta.get("px_per_m")
    area_m2 = round(sum(e.shape().area for e in items) / px ** 2, 1) if px and items else None
    # строки таблиц со словами из названия зоны (например DORMITORY -> WEST WING DORMITORY = 6,095 SF)
    rows = [{**r, "table": tb["title"]} for tb in sheet.get("tables", []) for r in tb.get("rows", [])
            if _words(best) and _words(best) <= _words(r["label"])]
    table_m2 = round(sum(r["value"] * (SQFT_M2 if r["unit"].lower() in ("sf", "sq ft", "sqft", "ft2") else 1.0)
                         for r in rows if r["unit"].lower() in ("sf", "sq ft", "sqft", "ft2", "m2", "m²")), 1) or None
    res = {"legend_label": best, "zones_on_plan": len(items),
           "area_m2": table_m2 or area_m2,
           "area_source": "таблица листа" if table_m2 else ("цвет на плане × масштаб" if area_m2 else None),
           "table_rows": rows, "table_area_m2": table_m2,
           "color_area_m2": area_m2, "color_area_note": "по цвету: только свободный пол, без мебели, стен и санузлов"}
    score, note = 0.8, None  # таблица прочитана ИИ — medium, пока нет второго источника
    if not items:
        score, note = 0.6, "этот цвет на плане не найден — подсветить нечего"
    elif area_m2 and table_m2:
        r = area_m2 / table_m2
        if 0.85 <= r <= 1.15:
            score = AGREE_FLOOR  # площадь по цвету и масштабу совпала с таблицей листа
        elif r > 1.15:
            score, note = DISAGREE_CAP, f"по цвету {area_m2} м² больше, чем в таблице ({table_m2} м²) — проверьте"
        else:
            note = f"по цвету {area_m2} м² (без мебели и стен), в таблице {table_m2} м²"
    elif not table_m2 and area_m2:
        score, note = 0.7, "в таблицах листа строки для этой зоны нет — площадь по цвету, без мебели и стен"
    elif not px:
        note = "масштаб неизвестен — площадь зоны не посчитана"
    return Answer(res, "m2" if res["area_m2"] else None, level(score), round(score, 2), ids,
                  [_ev(e) for e in items], note, action="highlight")


def rooms(scene: Scene, name: str | None = None) -> Answer:
    """Экспликация помещений: площадь (м²) и погонные метры плинтуса по каждой подписанной комнате + итоги.
    Площадь — подпись на плане, если есть; иначе геометрия. Уверенность — по согласию источников (fp/spaces.py)."""
    sp = scene.meta.get("spaces")
    if sp is None:
        return Answer(None, confidence="low", score=0.3,
                      note="помещения не посчитаны — загрузите план заново или нажмите «Авто-масштаб»")
    if not sp:
        return Answer([], confidence="low", score=0.3, note=scene.meta.get("spaces_note") or "помещений не найдено")
    rows = sp
    if name:
        w = _words(name) or {name.lower()}
        rows = [r for r in sp if w & _words(r["name"]) or name.lower() in r["name"].lower()]
        if not rows:
            return Answer([], confidence="low", score=0.3,
                          note=f"помещения {name!r} нет; есть: {', '.join(r['name'] for r in sp)}")
    ok = [r for r in rows if r.get("ok")]
    total = {"area_m2": round(sum(r["area_m2"] for r in ok), 2),
             "baseboard_m": round(sum(r["baseboard_m"] or 0 for r in ok), 2)}
    rank = {"high": 1.0, "medium": 0.8, "low": 0.5}
    score = min((rank[r["confidence"]] for r in ok), default=0.3)
    failed = [r["name"] for r in rows if not r.get("ok")]
    no_base = [r["name"] for r in ok if r["baseboard_m"] is None]
    notes = []
    if failed:
        notes.append(f"не посчитаны: {', '.join(failed)} (контур не замкнут)")
        score = min(score, 0.5)
    if no_base:
        notes.append(f"плинтус не посчитан (нет контура): {', '.join(no_base)}")
    rough = [r["name"] for r in ok if r.get("baseboard_note")]
    if rough:
        notes.append(f"неровный контур, плинтус может быть завышен: {', '.join(rough)}")
    low = [r["name"] for r in ok if r["confidence"] == "low" and not r.get("baseboard_note")]
    if low:
        notes.append(f"источники разошлись: {', '.join(low)}")
    ids = [r["id"] for r in ok if r.get("id")]
    return Answer({"rooms": rows, "total": total}, "m2/m", level(score), round(score, 2), ids,
                  [{"kind": "space", "id": r.get("id"), "name": r["name"], "source": r.get("area_source")} for r in ok],
                  "; ".join(notes) or None, action="highlight")


# --- описание инструментов для LLM (Claude tool calling) ---
_CLS_ENUM = CLASSES + ["room", "sauna"]
TOOLS = [
    {"name": "highlight", "description": "Подсветить на плане все объекты класса (опционально только в указанной комнате). Возвращает количество, id и уверенность.",
     "input_schema": {"type": "object", "properties": {
         "cls": {"type": "string", "enum": _CLS_ENUM, "description": "класс объектов; 'room' = все комнаты"},
         "room": {"type": "string", "description": "id комнаты (например bath_1) или тип (bath, kitchen, bedroom...)"}},
         "required": ["cls"], "additionalProperties": False}},
    {"name": "highlight_ids", "description": "Подсветить конкретные объекты/комнаты по id.",
     "input_schema": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                      "required": ["ids"], "additionalProperties": False}},
    {"name": "count", "description": "Посчитать объекты класса, опционально в комнате. Ничего не подсвечивает.",
     "input_schema": {"type": "object", "properties": {
         "cls": {"type": "string", "enum": _CLS_ENUM}, "room": {"type": "string"}},
         "required": ["cls"], "additionalProperties": False}},
    {"name": "list_rooms", "description": "Список комнат плана: id, тип, число объектов, уверенность.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "objects_in", "description": "Какие объекты находятся в комнате (по id или типу). Подсвечивает их.",
     "input_schema": {"type": "object", "properties": {"room": {"type": "string"}},
                      "required": ["room"], "additionalProperties": False}},
    {"name": "area", "description": "Площадь комнаты (по id или типу). В м² только если известен масштаб, иначе в пикселях².",
     "input_schema": {"type": "object", "properties": {"room": {"type": "string"}},
                      "required": ["room"], "additionalProperties": False}},
    {"name": "legend", "description": "Данные самого листа чертежа: штамп (название, номер листа, дата), масштаб, легенда цветов, "
                               "таблицы (количество комнат, площади), общие примечания, габаритные размеры. "
                               "Без label — всё это целиком: используй для вопросов о количестве комнат, площадях этажей/крыльев, "
                               "номере листа, примечаниях (ответ = строки таблиц). С label (название из легенды, "
                               "например 'DORMITORY') — подсветить зоны этого цвета и дать их площадь. "
                               "Зоны — это закрашенные области, а НЕ комнаты: число зон не равно числу комнат.",
     "input_schema": {"type": "object", "properties": {"label": {"type": "string"}}, "additionalProperties": False}},
    {"name": "rooms", "description": "Экспликация помещений: площадь в м² и погонные метры плинтуса для каждой подписанной "
                              "комнаты плана и итоги; подсвечивает их. name — отфильтровать по названию (например 'MH', 'WC', "
                              "'kitchen'); без name — все помещения. Используй для вопросов о площадях помещений и плинтусах.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False}},
    {"name": "clear", "description": "Снять всю подсветку с плана.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]
for _t in TOOLS:
    _t["strict"] = True

FUNCS = {"highlight": highlight, "highlight_ids": highlight_ids, "count": count, "list_rooms": list_rooms,
         "objects_in": objects_in, "area": area, "legend": legend, "rooms": rooms, "clear": clear}


def call(scene: Scene, name: str, args: dict) -> Answer:
    if name not in FUNCS:
        raise ValueError(f"нет такого инструмента: {name}")
    return FUNCS[name](scene, **args)
