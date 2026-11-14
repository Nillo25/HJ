"""Словарь обозначений помещений на планах (fp/glossary.json): 15 языков, полные названия и сокращения.

    lookup("MH 2 12.5 m²")      -> {"kind": "bedroom", "cls": "room_bedroom", "term": "MH", "lang": "fi", ...}
    room_class("Group Room")    -> "room_other"
    add_term("VH", "dressing")  -> дополнение пользователя (data/glossary_user.json), сразу участвует в поиске

Правила поиска:
- регистр и диакритика не важны («KEITTIÖ» = «keittio»), ё = е;
- полное название — последовательность слов подписи («entrance hall»); длинные слова (≥ 5 букв) ищутся и внутри
  составных слов («Elternschlafzimmer», «makuuhuone2»);
- сокращение — только целое слово подписи («K», «MH», «S»); однобуквенное — только первым словом
  («K 8.5», но не «stair B»);
- побеждает самое длинное совпадение: «Multi-Purpose Hall» — зал, а не прихожая (hall), «kitchen/dining» — кухня.
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).with_name("glossary.json")
USER = Path(__file__).resolve().parent.parent / "data" / "glossary_user.json"
_lock = threading.Lock()


def norm(s: str) -> str:
    """Нижний регистр, без диакритики (кроме кириллицы й/ё -> и/е — тоже без), знаки -> пробелы."""
    s = unicodedata.normalize("NFKD", str(s).lower().replace("ё", "е"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ß", "ss").replace("ø", "o").replace("æ", "ae").replace("ł", "l")
    return " ".join(re.findall(r"[^\W_]+", s))


def _merge(base: dict, extra: dict) -> dict:
    for kind, k in (extra.get("kinds") or {}).items():
        dst = base["kinds"].setdefault(kind, {"class": k.get("class", "room_other"), "terms": {}, "abbr": {}})
        for part in ("terms", "abbr"):
            for lang, vals in (k.get(part) or {}).items():
                lst = dst.setdefault(part, {}).setdefault(lang, [])
                lst.extend(v for v in vals if v not in lst)
    return base


@lru_cache(maxsize=1)
def _index() -> tuple[dict, list]:
    data = json.loads(BASE.read_text())
    if USER.exists():
        try:
            data = _merge(data, json.loads(USER.read_text()))
        except (OSError, ValueError):
            pass
    entries = []  # (норм. токены, вид, исходный текст, язык, это сокращение)
    for kind, k in data["kinds"].items():
        for part in ("terms", "abbr"):
            for lang, vals in (k.get(part) or {}).items():
                for v in vals:
                    t = norm(v).split()
                    if t:
                        entries.append((tuple(t), kind, v, lang, part == "abbr"))
    # длинные — первыми: при равной длине выигрывает более раннее в файле
    entries.sort(key=lambda e: -len(" ".join(e[0])))
    return data, entries


def reload() -> None:
    _index.cache_clear()


def kinds() -> dict:
    return _index()[0]["kinds"]


def _find(toks: tuple, is_abbr: bool, words: list, joined: str) -> int:
    """Позиция совпадения в joined (" w1 w2 … "), -1 — нет."""
    n = len(toks)
    if is_abbr:
        if n == 1 and len(toks[0]) == 1:      # однобуквенное — только первым словом
            return 0 if words[0] == toks[0] else -1
        for i in range(len(words) - n + 1):
            if tuple(words[i:i + n]) == toks:
                return len(" ".join(words[:i])) + (1 if i else 0)
        return -1
    phrase = " ".join(toks)
    p = joined.find(f" {phrase} ")
    if p >= 0:
        return p
    return joined.find(phrase) if len(phrase) >= 5 else -1


def lookup_all(label: str) -> list[dict]:
    """Все виды помещений в подписи по порядку («K/OH» -> кухня, гостиная). Вложенные совпадения отброшены:
    в «Entrance Hall» только прихожая, без отдельного «hall»."""
    words = []
    for w in norm(label).split():   # цифры отдельно от букв: «MH2» -> «mh 2»
        words.extend(re.findall(r"\d+|\D+", w))
    if not words:
        return []
    data, entries = _index()
    joined = " " + " ".join(words) + " "
    spans, out = [], []
    for toks, kind, text, lang, is_abbr in entries:       # длинные первыми
        p = _find(toks, is_abbr, words, joined)
        if p < 0:
            continue
        a, b = p, p + len(" ".join(toks)) + 1
        if any(a < y and x < b for x, y in spans):          # пересекается с уже найденным (более длинным)
            continue
        spans.append((a, b))
        k = data["kinds"][kind]
        out.append((a, {"kind": kind, "cls": k["class"], "term": text, "lang": lang, "abbr": is_abbr, "ru": k.get("ru")}))
    seen, res = set(), []
    for _, m in sorted(out, key=lambda t: t[0]):
        if m["kind"] not in seen:
            seen.add(m["kind"])
            res.append(m)
    return res


def lookup(label: str) -> dict | None:
    """Главный вид помещения по подписи (первый по порядку слов); None — в словаре нет."""
    ms = lookup_all(label)
    return ms[0] if ms else None


def room_class(label: str | None, default: str = "room_other") -> str:
    m = lookup(label) if label else None
    return m["cls"] if m else default


def add_term(term: str, kind: str, lang: str = "user", abbr: bool | None = None) -> dict:
    """Дополнение пользователя: «VH — гардеробная» -> add_term("VH", "dressing"). Хранится в data/glossary_user.json
    (не в репозитории), сразу участвует в поиске. abbr=None: сокращение, если ≤ 4 символов без пробелов."""
    if kind not in kinds():
        raise ValueError(f"нет такого вида помещения: {kind}; есть: {', '.join(kinds())}")
    term = term.strip()
    if not term:
        raise ValueError("пустое обозначение")
    if abbr is None:
        abbr = len(term) <= 4 and " " not in term
    with _lock:
        user = json.loads(USER.read_text()) if USER.exists() else {"kinds": {}}
        k = user["kinds"].setdefault(kind, {"terms": {}, "abbr": {}})
        lst = k.setdefault("abbr" if abbr else "terms", {}).setdefault(lang, [])
        if term not in lst:
            lst.append(term)
        USER.parent.mkdir(parents=True, exist_ok=True)
        USER.write_text(json.dumps(user, ensure_ascii=False, indent=1))
    reload()
    return lookup(term) or {}
