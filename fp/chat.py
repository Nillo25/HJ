"""Чат над планом: сначала дешёвый шаблонный разбор ($0, parsed_by=cache),
если не понял — Claude с tool calling (parsed_by=llm). Ответ всегда считает наш код (answered_by=db)."""
from __future__ import annotations

import json
import os
import re
import time

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
    best, best_len = None, 0
    for cls, words in SYNONYMS.items():
        for w in words:
            if w in text and len(w) > best_len:
                best, best_len = cls, len(w)
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
    if re.search(r"выдел|подсвет|покажи|показать|отмет|найди|где\b|highlight|show|find", t):
        return "highlight", args
    if re.search(r"сколько|посчит|количеств|how many|\bcount\b", t):
        return "count", args
    return None


def _format(name: str, args: dict, a: Answer) -> str:
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
    elif name == "area":
        if not a.answer:
            txt = "Комнату не нашёл."
        else:
            parts = [(f"{r['id']}: {r['area_m2']} м²" if r["area_m2"] is not None else f"{r['id']}: {r['area_px2']} px²")
                     + (f" (на плане подписано {r['label_m2']} м²)" if r.get("label_m2") else "")
                     for r in a.answer]
            txt = "Площадь — " + "; ".join(parts) + f".{conf}"
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


def chat(scene: Scene, message: str, history: list[dict] | None = None, use_llm: bool | None = None) -> dict:
    t0 = time.perf_counter()
    rule = parse_rule(message)
    # у листа есть напечатанные таблицы, а вопрос про этаж/лист/итоги — отвечать по таблицам (через legend),
    # а не подсчётом детекций (модель обучена на квартирах CubiCasa и на других листах даёт шум)
    sheet = scene.meta.get("sheet") or {}
    low = message.lower()
    if rule and rule[0] == "count" and sheet.get("tables") and \
            re.search(r"этаж|лист|таблиц|всего|итог|floor|sheet|total|table", low):
        rule = None
    # вопрос про легенду или со словом из неё (circulation, dormitory…) — к данным листа, не к детекциям
    legend_words = {w for lg in sheet.get("legend", []) for w in re.findall(r"[a-zа-яё]{4,}", lg["label"].lower())}
    if rule and rule[0] not in ("clear", "legend") and (re.search(r"легенд|legend", low) or
                                                        any(w in low for w in legend_words)):
        rule = None
    if rule:
        name, args = rule
        a = call(scene, name, args)
        return {"reply": _format(name, args, a), "action": _merge_actions([a]), "confidence": _worst([a]),
                "calls": [{"tool": name, "args": args, "result": a.to_dict()}],
                "parsed_by": "cache", "answered_by": "db", "cost_usd": 0.0,
                "latency_ms": round((time.perf_counter() - t0) * 1000)}

    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if not use_llm:
        return {"reply": "Не понял запрос, а LLM не подключена (нет ANTHROPIC_API_KEY). Примеры: "
                         "«выдели все туалеты», «сколько окон в спальне», «какие комнаты», "
                         "«площадь кухни», «очисти».",
                "action": {"type": "none"}, "confidence": None, "calls": [], "parsed_by": "none",
                "answered_by": None, "cost_usd": 0.0, "latency_ms": round((time.perf_counter() - t0) * 1000)}
    out = _chat_llm(scene, message, history or [])
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return out


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
    return (
        "You answer questions about one floor plan sheet. The plan is already recognised into a scene graph; "
        "you cannot see pixels. Get every number, id and confidence from the tools - never estimate or invent.\n"
        "- If the user asks to show/mark/highlight something, call highlight / highlight_ids / objects_in so it is "
        "highlighted on the plan; for 'how many' questions call count.\n"
        "- Reply in the SAME language as the user's last message (e.g. Estonian question -> Estonian answer).\n"
        "- Keep it to one or two short sentences with the number. Do not mention scores, element ids or tool names "
        "unless the user asks.\n"
        "- If the tool result has confidence low, add a short request to check the result on the plan.\n"
        f"Plan summary: objects {json.dumps(counts, ensure_ascii=False)}; rooms: {rooms}.{sheet_line}"
    )


def _chat_llm(scene: Scene, message: str, history: list[dict]) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    msgs = [{"role": h["role"], "content": h["content"]} for h in history[-10:]
            if h.get("role") in ("user", "assistant") and h.get("content")]
    msgs.append({"role": "user", "content": message})
    results: list[Answer] = []
    calls: list[dict] = []
    tin = tout = 0
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
            tool_results.append({"type": "tool_result", "tool_use_id": u.id, "content": content, "is_error": err})
        msgs.append({"role": "user", "content": tool_results})
    else:
        reply = reply or "Не уложился в лимит шагов."
    return {"reply": reply, "action": _merge_actions(results), "confidence": _worst(results), "calls": calls,
            "parsed_by": "llm", "answered_by": "db",
            "model": MODEL, "cost_usd": round((tin * PRICES.get(MODEL, (4.0, 20.0))[0]
                                               + tout * PRICES.get(MODEL, (4.0, 20.0))[1]) / 1e6, 6)}
