"""Кэш чата в SQLite: повтор вопроса не стоит денег.

Два уровня:
- answers   — (версия сцены, вопрос) -> готовый ответ. Версия = sha256 содержимого сцены и кода ответов, поэтому
              любая правка сцены (масштаб, повторный разбор) или подсчёта сама делает старые ответы недействительными.
- templates — (вид листа, вопрос) -> какие инструменты вызвать. Найдено LLM на одном плане, повторяется на
              любом другом без LLM: ответ считает наш код по сцене этого плана.

По умолчанию выключен; сервер включает его через configure(путь).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

_path: Path | None = None
_lock = threading.Lock()


def configure(path: Path | str | None):
    global _path
    _path = Path(path) if path else None
    if _path:
        _path.parent.mkdir(parents=True, exist_ok=True)
        with _conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS answers (scene TEXT, q TEXT, response TEXT, created REAL,"
                      " hits INTEGER DEFAULT 0, PRIMARY KEY (scene, q))")
            c.execute("CREATE TABLE IF NOT EXISTS templates (kind TEXT, q TEXT, calls TEXT, lang TEXT, created REAL,"
                      " hits INTEGER DEFAULT 0, PRIMARY KEY (kind, q))")


def enabled() -> bool:
    return _path is not None


def _conn():
    return sqlite3.connect(_path, timeout=5)


def normalize(message: str) -> str:
    """Регистр, ё, пробелы и пунктуация по краям не меняют смысл вопроса."""
    t = message.lower().replace("ё", "е")
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip(" ?!.,;:¿¡")


_FOLLOWUP = re.compile(r"^(а|и|а если|а теперь|а там|тогда|ещё|еще|and|what about|how about|and in|also|same)\b")


def is_followup(q: str) -> bool:
    """«а в спальне?», «and the bedroom?» — смысл зависит от предыдущих реплик, кэшу не доверяем."""
    return bool(_FOLLOWUP.match(q)) or len(q.split()) <= 1


def _code_version() -> str:
    """Отпечаток кода, который считает ответы: правка подсчёта или маршрутов делает старые ответы недействительными
    (иначе тот же вопрос к той же сцене вернул бы ответ, посчитанный прежним кодом)."""
    root = Path(__file__).parent
    h = hashlib.sha256()
    for name in ("query.py", "chat.py", "route.json"):
        f = root / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


CODE_VERSION = _code_version()


def scene_key(scene) -> str:
    body = json.dumps(scene.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{CODE_VERSION}|{body}".encode()).hexdigest()


def scene_kind(scene) -> str:
    """Шаблон, выученный на листе с таблицами/легендой, не переносится на простой план и наоборот:
    чат выбирает инструменты по-разному."""
    sheet = scene.meta.get("sheet") or {}
    return "sheet" if sheet.get("tables") or sheet.get("legend") else "plan"


def get_answer(skey: str, q: str) -> dict | None:
    if not _path:
        return None
    with _lock, _conn() as c:
        row = c.execute("SELECT response FROM answers WHERE scene=? AND q=?", (skey, q)).fetchone()
        if row:
            c.execute("UPDATE answers SET hits = hits + 1 WHERE scene=? AND q=?", (skey, q))
    return json.loads(row[0]) if row else None


def put_answer(skey: str, q: str, response: dict):
    if not _path:
        return
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO answers (scene, q, response, created) VALUES (?, ?, ?, ?)",
                  (skey, q, json.dumps(response, ensure_ascii=False), time.time()))


def get_template(kind: str, q: str) -> tuple[list[dict], str] | None:
    if not _path:
        return None
    with _lock, _conn() as c:
        row = c.execute("SELECT calls, lang FROM templates WHERE kind=? AND q=?", (kind, q)).fetchone()
        if row:
            c.execute("UPDATE templates SET hits = hits + 1 WHERE kind=? AND q=?", (kind, q))
    return (json.loads(row[0]), row[1]) if row else None


def put_template(kind: str, q: str, calls: list[dict], lang: str):
    if not _path:
        return
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO templates (kind, q, calls, lang, created) VALUES (?, ?, ?, ?, ?)",
                  (kind, q, json.dumps(calls, ensure_ascii=False), lang, time.time()))


def stats() -> dict:
    if not _path:
        return {"enabled": False}
    with _lock, _conn() as c:
        a = c.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM answers").fetchone()
        t = c.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM templates").fetchone()
    return {"enabled": True, "answers": a[0], "answer_hits": a[1], "templates": t[0], "template_hits": t[1]}
