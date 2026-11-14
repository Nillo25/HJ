"""Чат над планом: сначала дешёвый шаблонный разбор ($0, parsed_by=cache),
если не понял — Claude с tool calling (parsed_by=llm). Ответ всегда считает наш код (answered_by=db)."""
from __future__ import annotations

import json
import os
import re
import time

from . import cache
from .classes import LABELS_RU, SYNONYMS
from .query import TOOLS, Answer, call
from .scene import Scene

# Разбор вопроса — дешёвая модель (Stoa: цена ответа — 25% оценки). Opus — через FP_CLAUDE_MODEL=claude-opus-5-5
MODEL = os.environ.get("FP_CLAUDE_MODEL", "claude-haiku-5-5")
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}  # $/1M ток.

ROOM_WORDS = {  # слово после «в/на» -> тип комнаты
    "ванн": "bath", "санузл": "bath", "санузел": "bath", "туалет": "bath", "душев": "bath",
    "кухн": "kitchen", "спальн": "bedroom", "гостин": "living", "зал": "living",
    "прихож": "entry", "коридор": "entry", "холл": "entry", "кладов": "storage",
    "гардероб": "storage", "балкон": "outdoor", "террас": "outdoor", "лоджи": "outdoor",
}


def _room_ref(text: str) -> tuple[str | None, tuple[int, int] | None]:
    """«в санузле», «на кухне», «в bath_1» -> (room, span)."""
    for m in re.finditer(r"\b(?:в|во|на|in)\s+([\w]+)", text):
        w = m.group(1)
        if re.fullmatch(r"[a-z]+_\d+", w):
            return w, m.span()
        for stem, t in ROOM_WORDS.items():
            if w.startswith(stem):
                return t, m.span()
    return None, None


def _find_class(text: str) -> str | None:
    """Класс по ПЕРВОМУ упоминанию, при равенстве — по самому длинному слову. В длинных вопросах
    («How many sinks and wash basins (kitchen and bathroom combined)…») предмет вопроса стоит первым,
    а комнаты в скобках — уточнения; раньше побеждало самое длинное слово, и вопрос о раковинах
    превращался в подсчёт санузлов."""
    best, key = None, None
    t = text.replace("floor plan", "floor_plan")  # «план этажа» — не вопрос об этажах
    for cls, words in SYNONYMS.items():
        for w in words:
            i = t.find(w)
            if i >= 0 and (key is None or (i, -len(w)) < key):
                best, key = cls, (i, -len(w))
    return best


def parse_rule(message: str) -> tuple[str, dict] | None:
    t = message.lower().replace("ё", "е")
    if re.search(r"очист|сброс|сними|убери подсвет|\bclear\b|\breset\b", t):
        return "clear", {}
    if re.search(r"(какие|список|перечисл).{0,12}(комнат|помещен)|list rooms", t):
        return "list_rooms", {}
    if re.search(r"легенд|штамп|\blegend\b", t) and not re.search(r"площад|сколько|выдел|покажи", t):
        return "legend", {}
    if re.search(r"плинтус|baseboard|skirting|экспликац|площад\w* (всех |каждого )?(помещени|комнат)|"
                 r"(floor|room) areas", t):
        # «плинтус в спальнях / in the bedrooms (MH)» — с фильтром по помещению разбирает LLM
        return None if re.search(r"\b(в|во|in|for|of the)\s+\w{2,}|\(", t) else ("rooms", {})

    # лист трубопроводов: длина труб по диаметрам, арматура по маркам
    if re.search(r"труб|трубопровод|\bpipe|piping", t) and not re.search(r"опор|support", t):
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:\"|''|дюйм|inch|in\b|-?inch)", t)
        return "pipes", ({"size": float(m.group(1).replace(",", "."))} if m else {})
    m = re.search(r"\b(cv|fscv|bpcv|mav|pg|pit|dpt|fsi|fsr|fp)\b", t)
    if m and re.search(r"сколько|how many|покажи|выдел|show|count|highlight", t):
        return "fittings", {"tag": m.group(1).upper()}
    for words, kind in ((r"обратн\w* клапан|check valve", "CV"), (r"манометр|pressure gauge", "PG"),
                        (r"воздухоотвод|air vent", "MAV"), (r"насос\w*|pump", "FP"),
                        (r"клапан|задвижк|арматур|valve", "valve"), (r"манометр|прибор|датчик|gauge|instrument", "instrument"),
                        (r"опор\w* труб|pipe support|опор", "support")):
        if re.search(words, t) and re.search(r"сколько|how many|покажи|выдел|show|count|highlight|какие|list", t):
            return "fittings", {"tag": kind}

    room, span = _room_ref(t)
    rest = t[:span[0]] + t[span[1]:] if span else t
    cls = _find_class(rest)

    if re.search(r"площад|\barea\b", t):
        target = room or (cls.removeprefix("room_") if cls and cls.startswith("room_") else None)
        return ("area", {"room": target}) if target else None
    if re.search(r"что (есть |находится |стоит )?(в|на)\b|what.s in", t) and room and not cls:
        return "objects_in", {"room": room}
    if cls is None:
        return None
    args = {"cls": cls}
    if room:
        args["room"] = room
    # «покажи … и сколько» — подсветка: она тоже возвращает число, а план подсвечен
    if cls not in ("floor", "wc_room") and \
            re.search(r"выдел|подсвет|покажи|показать|отмет|найди|где\b|highlight|show|find", t):
        return "highlight", args
    if re.search(r"сколько|посчит|количеств|how many|\bcount\b", t):
        return "count", args
    return None


def _lang(message: str) -> str | None:
    """Язык ответа для шаблона: ru, en или None (другой язык — шаблоном не отвечаем, только LLM)."""
    if re.search(r"[а-яё]", message.lower()):
        return "ru"
    words = set(re.findall(r"[a-z]+", message.lower()))
    if re.fullmatch(r"[\x00-\x7f]*", message) and words & {
            "how", "many", "what", "where", "which", "show", "highlight", "count", "list", "the", "is", "are",
            "area", "rooms", "room", "find", "mark", "total", "number", "clear", "legend", "sheet", "there"}:
        return "en"
    return None


def _format_en(name: str, args: dict, a: Answer) -> str:
    conf = {"high": "", "medium": " (medium confidence)", "low": " (low confidence - please check on the plan)"}[a.confidence]
    what = (args.get("cls") or "").replace("room_", "").replace("_", " ")
    where = f" in {args['room']}" if args.get("room") else ""
    if name == "clear":
        return "Highlighting cleared."
    if name in ("highlight", "count"):
        return f"{'Highlighted' if name == 'highlight' else 'Found'} {what}{where}: {a.answer}.{conf}"
    if name == "list_rooms":
        return f"Rooms: {', '.join(r['id'] for r in a.answer) or 'none found'}.{conf}"
    if name == "objects_in":
        items = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in a.answer.items()) if a.answer else "nothing"
        return f"In {args['room']}: {items}.{conf}"
    if name == "legend" and isinstance(a.answer, dict) and "tables" in a.answer:
        tb = a.answer.get("title_block") or {}
        head = " · ".join(v for v in (tb.get("drawing_title"), tb.get("sheet_number"), tb.get("project")) if v)
        leg = "; ".join(f"{x['label']} (on plan: {x['zones_on_plan']})" for x in a.answer["legend"]) or "none"
        tabs = "; ".join(f"{t['title']}: {len(t['rows'])} rows" for t in a.answer["tables"]) or "none"
        return f"Sheet: {head or '-'}. Scale: {a.answer.get('scale_text') or '-'}.\nLegend: {leg}.\nTables: {tabs}.{conf}"
    if name == "rooms" and isinstance(a.answer, dict):
        lines = [(f"{r['name']}: {r['area_m2']} m²" + (f", baseboard {r['baseboard_m']} m" if r["baseboard_m"] is not None else ""))
                 if r.get("ok") else f"{r['name']}: not measured" for r in a.answer["rooms"]]
        tot = a.answer["total"]
        return "\n".join(lines) + f"\nTotal: {tot['area_m2']} m², baseboard {tot['baseboard_m']} m.{conf}"
    if name == "pipes" and isinstance(a.answer, dict):
        parts = ", ".join(f"{k}: {v} m" for k, v in a.answer["by_size_m"].items())
        return f"Pipe length by size: {parts}; total {a.answer['total_m']} m.{conf}"
    if name == "fittings" and isinstance(a.answer, dict):
        return "Tags on the sheet: " + ", ".join(f"{k} × {v['count']}" for k, v in a.answer.items()) + f".{conf}"
    if name == "area":
        if not a.answer:
            return "Room not found."
        parts = [f"{r['id']}: {r['area_m2']} m²" if r["area_m2"] is not None else f"{r['id']}: {r['area_px2']} px²"
                 for r in a.answer]
        return "Area - " + "; ".join(parts) + f".{conf}"
    return json.dumps(a.answer, ensure_ascii=False)


def _format(name: str, args: dict, a: Answer, lang: str = "ru") -> str:
    if lang == "en":
        return _format_en(name, args, a)
    conf = {"high": "", "medium": " (уверенность средняя)", "low": " (уверенность низкая — проверьте)"}[a.confidence]
    what = LABELS_RU.get(args.get("cls", ""), "комнаты" if args.get("cls") == "room" else args.get("cls", ""))
    where = f" в {args['room']}" if args.get("room") else ""
    if name == "clear":
        return "Подсветку снял."
    if name in ("highlight", "count"):
        verb = "Подсветил" if name == "highlight" else "Нашёл"
        txt = f"{verb} {what}{where}: {a.answer} шт.{conf}"
    elif name == "list_rooms":
        rooms = ", ".join(f"{r['id']} ({r['name_ru']})" for r in a.answer) or "комнат не найдено"
        txt = f"Комнаты: {rooms}.{conf}"
    elif name == "objects_in":
        items = ", ".join(f"{LABELS_RU.get(k, k)}: {v}" for k, v in a.answer.items()) if a.answer else "ничего"
        txt = f"В {args['room']}: {items}.{conf}"
    elif name == "legend" and isinstance(a.answer, dict) and "tables" in a.answer:
        tb = a.answer.get("title_block") or {}
        head = " · ".join(v for v in (tb.get("drawing_title"), tb.get("sheet_number"), tb.get("project")) if v)
        leg = "; ".join(f"{x['label']} (на плане: {x['zones_on_plan']})" for x in a.answer["legend"]) or "нет"
        tabs = "; ".join(f"{t['title']}: {len(t['rows'])} строк" for t in a.answer["tables"]) or "нет"
        txt = f"Лист: {head or '—'}. Масштаб: {a.answer.get('scale_text') or '—'}.\nЛегенда: {leg}.\nТаблицы: {tabs}.{conf}"
    elif name == "rooms" and isinstance(a.answer, dict):
        lines = []
        for r in a.answer["rooms"]:
            if not r.get("ok"):
                lines.append(f"{r['name']}: не посчитано")
                continue
            base = f", плинтус {r['baseboard_m']} м" if r["baseboard_m"] is not None else ""
            lines.append(f"{r['name']}: {r['area_m2']} м² ({r['area_source']}){base}")
        tot = a.answer["total"]
        txt = "\n".join(lines) + f"\nИтого: {tot['area_m2']} м², плинтус {tot['baseboard_m']} м.{conf}"
    elif name == "pipes" and isinstance(a.answer, dict):
        parts = ", ".join(f"{k} — {v} м" for k, v in a.answer["by_size_m"].items())
        txt = f"Трубы по диаметрам: {parts}; всего {a.answer['total_m']} м.{conf}"
    elif name == "fittings" and isinstance(a.answer, dict):
        txt = "По маркам на листе: " + ", ".join(f"{k} × {v['count']}" for k, v in a.answer.items()) + f".{conf}"
    elif name == "area":
        if not a.answer:
            txt = "Комнату не нашёл."
        else:
            parts = [(f"{r['id']}: {r['area_m2']} м²" if r["area_m2"] is not None else f"{r['id']}: {r['area_px2']} px²")
                     + (f" (на плане подписано {r['label_m2']} м²)" if r.get("label_m2") else "")
                     for r in a.answer]
            txt = "Площадь — " + "; ".join(parts) + f".{conf}"
    elif a.answer is None and a.note:  # ответа нет — только объяснение почему, без «null»
        return a.note[:1].upper() + a.note[1:] + "."
    else:
        txt = json.dumps(a.answer, ensure_ascii=False)
    if a.note:
        txt += f"\nПримечание: {a.note}."
    return txt


def _merge_actions(results: list[Answer]) -> dict:
    """Итоговое действие для фронта: последний clear сбрасывает, highlight-и копятся."""
    ids: list[str] = []
    cleared = False
    for a in results:
        if a.action == "clear":
            ids, cleared = [], True
        elif a.action == "highlight":
            ids += [i for i in a.ids if i not in ids]
    if ids:
        return {"type": "highlight", "ids": ids}
    return {"type": "clear"} if cleared else {"type": "none"}


def _worst(results: list[Answer]) -> str | None:
    order = {"low": 0, "medium": 1, "high": 2}
    rs = [a for a in results if a.action != "clear"]
    return min((a.confidence for a in rs), key=order.get) if rs else None


def _space_name(scene: Scene, message: str) -> str | None:
    """Название помещения из экспликации, которое целиком есть в вопросе (самое длинное). Короткие (< 5 букв:
    «WC», «Lift») — только если вопрос и есть это название, чтобы «сколько WC» оставалось подсчётом."""
    t = " " + re.sub(r"[^\w\s-]", " ", message.lower().replace("ё", "е")) + " "
    best = None
    for sp in scene.meta.get("spaces") or []:
        n = (sp.get("name") or "").lower().replace("ё", "е").strip()
        if not n:
            continue
        hit = f" {n} " in t if len(n) >= 5 else t.strip() == n
        if hit and (best is None or len(n) > len(best)):
            best = sp["name"]
    return best


def chat(scene: Scene, message: str, history: list[dict] | None = None, use_llm: bool | None = None) -> dict:
    t0 = time.perf_counter()
    rule = parse_rule(message)
    # у листа есть напечатанные таблицы, а вопрос про этаж/лист/итоги — отвечать по таблицам (через legend),
    # а не подсчётом детекций (модель обучена на квартирах CubiCasa и на других листах даёт шум)
    sheet = scene.meta.get("sheet") or {}
    low = message.lower()
    if rule and rule[0] == "count" and rule[1].get("cls") != "floor" and sheet.get("tables") and \
            re.search(r"этаж|лист|таблиц|всего|итог|floor(?! plan)|sheet|total|table", low):
        rule = None
    # вопрос про легенду или со словом из неё (circulation, dormitory…) — к данным листа, не к детекциям
    legend_words = {w for lg in sheet.get("legend", []) for w in re.findall(r"[a-zа-яё]{4,}", lg["label"].lower())}
    if rule and rule[0] not in ("clear", "legend") and (re.search(r"легенд|legend", low) or
                                                        any(w in low for w in legend_words)):
        rule = None
    # название помещения из экспликации («Multi-Purpose Hall», «Staff Office») — показать его, с площадью
    # и плинтусом; правила иначе видят в «Hall» класс «прихожая», а ИИ искал название в легенде
    sp_name = _space_name(scene, message)
    if sp_name and (rule is None or (rule[0] in ("count", "highlight", "area") and
                                     str(rule[1].get("cls") or rule[1].get("room") or "").replace("room_", "")
                                     in ("entry", "bath", "kitchen", "bedroom", "living", "storage", "outdoor", "other", "room", ""))):
        rule = ("rooms", {"name": sp_name})
    if rule:
        name, args = rule
        a = call(scene, name, args)
        return {"reply": _format(name, args, a, _lang(message) or "ru"), "action": _merge_actions([a]), "confidence": _worst([a]),
                "calls": [{"tool": name, "args": args, "result": a.to_dict()}],
                "parsed_by": "cache", "answered_by": "db", "cost_usd": 0.0,
                "latency_ms": round((time.perf_counter() - t0) * 1000)}

    # кэш: тот же вопрос к той же сцене — готовый ответ; тот же вопрос к другому плану — те же инструменты
    q = cache.normalize(message)
    # уточнения («а в спальне?») зависят от предыдущих реплик: не берём из кэша и не кладём в него
    standalone = not cache.is_followup(q)
    skey = kind = None
    if cache.enabled() and standalone:
        skey, kind = cache.scene_key(scene), cache.scene_kind(scene)
        hit = cache.get_answer(skey, q)
        if hit:
            return {**hit, "parsed_by": "answer_cache", "saved_usd": hit.get("cost_usd", 0.0), "cost_usd": 0.0,
                    "latency_ms": round((time.perf_counter() - t0) * 1000)}
        tpl = cache.get_template(kind, q)
        if tpl:
            out = _run_template(scene, *tpl)
            if out:
                out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
                return out

    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if not use_llm:
        return {"reply": "Не понял запрос, а LLM не подключена (нет ANTHROPIC_API_KEY). Примеры: "
                         "«выдели все туалеты», «сколько окон в спальне», «какие комнаты», "
                         "«площадь кухни», «очисти».",
                "action": {"type": "none"}, "confidence": None, "calls": [], "parsed_by": "none",
                "answered_by": None, "cost_usd": 0.0, "latency_ms": round((time.perf_counter() - t0) * 1000)}
    out = _chat_llm(scene, message, history or [])
    errors = out.pop("_errors", 0)
    if skey and out["calls"] and not errors:
        cache.put_answer(skey, q, out)
        lang = _lang(message)
        if lang and _templatable(out["calls"]):
            cache.put_template(kind, q, [{"tool": c["tool"], "args": c["args"]} for c in out["calls"]], lang)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return out


_TEMPLATE_TOOLS = {"count", "highlight", "list_rooms", "objects_in", "area", "rooms", "clear", "legend", "pipes", "fittings"}


def _templatable(calls: list[dict]) -> bool:
    """Шаблон переносится на другой план, только если аргументы не ссылаются на id этого плана
    (bedroom_2, sp3, zone1) и у каждого инструмента есть готовая формулировка ответа."""
    if not 1 <= len(calls) <= 3:
        return False
    for c in calls:
        if c["tool"] not in _TEMPLATE_TOOLS or (c["tool"] == "legend" and c["args"].get("label")):
            return False
        if any(isinstance(v, str) and re.fullmatch(r"[a-z]+_?\d+", v) for v in c["args"].values()):
            return False
    return True


def _run_template(scene: Scene, calls: list[dict], lang: str) -> dict | None:
    results, done = [], []
    try:
        for c in calls:
            a = call(scene, c["tool"], dict(c["args"]))
            results.append(a)
            done.append({"tool": c["tool"], "args": c["args"], "result": a.to_dict()})
    except Exception:  # noqa: BLE001 — шаблон не подошёл этой сцене, пусть разбирает LLM
        return None
    reply = "\n".join(_format(c["tool"], c["args"], a, lang) for c, a in zip(calls, results))
    if lang == "en" and _worst(results) == "low" and "check" not in reply:
        reply += " Please check on the plan."
    return {"reply": reply, "action": _merge_actions(results), "confidence": _worst(results), "calls": done,
            "parsed_by": "template", "answered_by": "db", "cost_usd": 0.0}


def _system_prompt(scene: Scene) -> str:
    counts: dict[str, int] = {}
    for o in scene.objects:
        counts[o.cls] = counts.get(o.cls, 0) + 1
    rooms = ", ".join(f"{r.id}" for r in scene.rooms) or "нет"
    sheet = scene.meta.get("sheet")
    sheet_line = ""
    if sheet:
        tb = sheet.get("title_block") or {}
        sheet_line = (
            f"\nSheet (read once at upload): '{tb.get('drawing_title', '')}' sheet {tb.get('sheet_number', '')}; "
            f"legend: {[lg['label'] for lg in sheet.get('legend', [])]}; "
            f"tables: {[t['title'] for t in sheet.get('tables', [])]}. "
            "For anything printed on the sheet (room counts, square footage, notes, title block) call legend() and "
            "answer from its tables; to show a legend colour on the plan call legend(label). "
            "Detected 'zone' objects are coloured areas, not rooms.")
    names = sorted({sp["name"] for sp in scene.meta.get("spaces") or [] if sp.get("name")})
    spaces_line = ""
    if names:
        spaces_line = (f"\nNamed rooms on the sheet (room schedule): {names[:80]}. For a named room (show it, its "
                       "area or baseboard) call rooms(name) with that name — not legend and not list_rooms "
                       "(list_rooms are the model's unnamed rooms).")
    return (
        "You answer questions about one floor plan sheet. The plan is already recognised into a scene graph; "
        "you cannot see pixels. Get every number, id and confidence from the tools - never estimate or invent.\n"
        "- If the user asks to show/mark/highlight something, call highlight / highlight_ids / objects_in so it is "
        "highlighted on the plan; for 'how many' questions call count.\n"
        "- Reply in the SAME language as the user's last message (e.g. Estonian question -> Estonian answer).\n"
        "- Keep it to one or two short sentences with the number. Do not mention scores, element ids or tool names "
        "unless the user asks.\n"
        "- If the tool result has confidence low, add a short request to check the result on the plan.\n"
        f"Plan summary: objects {json.dumps(counts, ensure_ascii=False)}; rooms: {rooms}.{sheet_line}{spaces_line}"
    )


def _chat_llm(scene: Scene, message: str, history: list[dict]) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    msgs = [{"role": h["role"], "content": h["content"]} for h in history[-10:]
            if h.get("role") in ("user", "assistant") and h.get("content")]
    msgs.append({"role": "user", "content": message})
    results: list[Answer] = []
    calls: list[dict] = []
    tin = tout = n_err = 0
    reply = ""
    for _ in range(6):
        kw = dict(model=MODEL, max_tokens=4000, system=_system_prompt(scene), tools=TOOLS, messages=msgs,
                  output_config={"effort": "low"})
        if MODEL.startswith("claude-haiku"):
            # выбор функции не требует рассуждений: без thinking быстрее и дешевле (допустимо при effort <= high);
            # серверного fallback у Haiku 5.5 нет — обычный messages.create
            resp = client.messages.create(**kw, thinking={"type": "disabled"})
        else:
            resp = client.beta.messages.create(**kw, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        tin += resp.usage.input_tokens
        tout += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            reply = "Модель отказалась отвечать на этот запрос."
            break
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            reply = "".join(b.text for b in resp.content if b.type == "text")
            break
        msgs.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for u in uses:
            try:
                a = call(scene, u.name, dict(u.input))
                results.append(a)
                calls.append({"tool": u.name, "args": dict(u.input), "result": a.to_dict()})
                content, err = json.dumps(a.to_dict(), ensure_ascii=False), False
            except Exception as e:  # noqa: BLE001 — ошибку отдаём модели
                content, err = f"Ошибка: {e}", True
                n_err += 1
            tool_results.append({"type": "tool_result", "tool_use_id": u.id, "content": content, "is_error": err})
        msgs.append({"role": "user", "content": tool_results})
    else:
        reply = reply or "Не уложился в лимит шагов."
    return {"reply": reply, "action": _merge_actions(results), "confidence": _worst(results), "calls": calls,
            "parsed_by": "llm", "answered_by": "db", "_errors": n_err,
            "model": MODEL, "cost_usd": round((tin * PRICES.get(MODEL, (4.0, 20.0))[0]
                                               + tout * PRICES.get(MODEL, (4.0, 20.0))[1]) / 1e6, 6)}
