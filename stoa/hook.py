"""Opt-in bridge from the HJ server chat to the stoa counting pipeline (FP_STOA=1).

Only Stoa-style whole-plan count questions are taken ("How many windows are shown on this floor plan?",
Finnish "Kuinka monta ..."); questions about a specific room, highlights, areas etc. stay with HJ's chat.
The original upload is kept next to the scene as src_<plan_id>.<ext> (server/app.py), because all stoa
accuracy numbers are measured on original images, not on HJ's preprocessed copy.
"""
import re
import time
from pathlib import Path

from .qtypes import qtype_of

_COUNT = re.compile(r"^\s*(how many|kuinka monta)\b", re.I)
_ROOM_REF = re.compile(r"\b(in the|in this room|inside|huoneessa|в )\b", re.I)


def is_stoa_question(message: str) -> str | None:
    if not _COUNT.search(message) or _ROOM_REF.search(message):
        return None
    try:
        return qtype_of(message)
    except ValueError:
        return None


def answer(scene_dir: Path, message: str) -> dict | None:
    qtype = is_stoa_question(message)
    src = next(iter(sorted(Path(scene_dir).glob("src_*"))), None)
    if not qtype or src is None:
        return None
    from . import router
    t0 = time.perf_counter()
    r = router.answer(src, qtype)
    en = message.strip().lower().startswith("how")
    flag = "" if r["plan_check"]["reliable"] else (
        " Plan flagged as unreliable: sources disagree, please check." if en
        else " Suunnitelma merkitty epävarmaksi: lähteet ovat eri mieltä, tarkista.")
    reply = (f"{r['answer']} (confidence: {r['confidence']}).{flag}" if en
             else f"{r['answer']} (varmuus: {r['confidence']}).{flag}")
    return {"reply": reply, "answer": r["answer"], "action": {"type": "none"}, "confidence": r["confidence"],
            "calls": [{"tool": "stoa_count", "args": {"qtype": qtype}, "result": r}],
            "parsed_by": r.get("parsed_by", "cache"), "answered_by": f"stoa:{r.get('answered_by')}",
            "evidence_original_px": r.get("evidence", []), "plan_check": r["plan_check"],
            "cost_usd": r.get("cost_usd", 0.0), "latency_ms": round((time.perf_counter() - t0) * 1000)}
