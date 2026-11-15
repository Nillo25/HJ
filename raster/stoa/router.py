"""Hybrid router: per question type, use the method that won on the bench.

ROUTE values:
  "opus" - Claude (CLAUDE_MODEL) one batch call per image, counts only; high confidence + YOLO boxes as evidence
           when YOLO gives the same count (or has no such class), medium otherwise
  "hj"   - colleague YOLO detector (hj_yolo), $0, YOLO's own calibrated confidence
CubiCasa is not used.
"""
import os
import time
from pathlib import Path

from . import claude_vlm

ROUTE = {
    "windows": "hj", "doors": "hj",
    "toilets": "opus", "sinks": "opus", "showers": "opus", "saunas": "opus", "kitchens": "opus",
    "bedrooms": "opus", "wc_rooms": "opus", "bathrooms": "opus", "balconies": "opus", "floors": "opus",
}
HJ_LEVEL = {"high": 0.95, "medium": 0.75, "low": 0.5}


def _level(c):
    return "high" if c >= 0.9 else "medium" if c >= 0.7 else "low"


# Plan-level reliability ("брак"): Claude and colleague YOLO disagree on many question types -> the plan is hard,
# answers on it are unreliable.
UNRELIABLE_DISAGREEMENT = float(os.environ.get("STOA_UNRELIABLE_DISAGREEMENT", "0.40"))
_plan_cache: dict = {}


def plan_check(path: Path, model: str | None = None) -> dict:
    """Disagreement of YOLO with our answers over the question types it covers (0 = all agree, 1 = none)."""
    from .qtypes import QTYPES
    from . import hj_yolo
    path = Path(path)
    key = (str(path), model)
    if key in _plan_cache:
        return _plan_cache[key]
    hj = hj_yolo.detect(path)["counts"]
    per, disagree = [], []
    for qt in QTYPES:
        if qt not in hj:
            continue
        a = _answer_raw(path, qt, model)["answer"]
        d = float(hj[qt] != a)
        per.append(d)
        if d > 0:
            disagree.append(qt)
    score = round(sum(per) / len(per), 3) if per else 0.0
    res = {"reliable": score <= UNRELIABLE_DISAGREEMENT, "disagreement": score, "disagree_types": disagree}
    _plan_cache[key] = res
    return res


def answer(path: Path, qtype: str, model: str | None = None) -> dict:
    """Answer + plan-level reliability; on an unreliable plan 'high' confidence is capped at 'medium'."""
    r = dict(_answer_raw(path, qtype, model))
    pc = plan_check(path, model)
    r["plan_check"] = pc
    if not pc["reliable"] and r["confidence"] == "high":
        r["confidence"] = "medium"
        r["confidence_score"] = min(r.get("confidence_score", 0.75), 0.75)
    return r


def _answer_raw(path: Path, qtype: str, model: str | None = None) -> dict:
    t0 = time.time()
    path = Path(path)
    route = ROUTE[qtype]
    if route == "hj":
        from . import hj_yolo
        d = hj_yolo.detect(path)
        conf = HJ_LEVEL.get(d["levels"][qtype], 0.5)
        return {"answer": d["counts"][qtype], "unit": "count", "confidence": _level(conf), "confidence_score": conf,
                "parsed_by": "cache", "answered_by": "detector", "route": route,
                "evidence": [{"kind": "bbox", "page": 1, "box": b["box"], "label": b["label"], "score": b["score"]}
                             for b in d["boxes"][qtype]],
                "cost_usd": 0.0, "call_cost_shared": False, "latency_ms": int((time.time() - t0) * 1000)}
    if route == "opus":
        rec = claude_vlm.call(path, claude_vlm.prompt_batch(boxes=False), claude_vlm.schema_batch(boxes=False), model)
        n = int(rec["data"]["counts"].get(qtype, 0) or 0)
        # Confidence = agreement with the independent YOLO detector (80 plans, from cache):
        # agree 95.5% correct (313), YOLO has no such class (floors, wc_rooms) 99% (117), disagree 67% (139).
        from . import hj_yolo
        d = hj_yolo.detect(path)
        has_hj = qtype in d["counts"]
        agree = has_hj and d["counts"][qtype] == n
        conf = 0.95 if agree or not has_hj else 0.75
        ev = [{"kind": "bbox", "page": 1, "box": b["box"], "label": b["label"], "score": b["score"]}
              for b in d["boxes"][qtype]] if agree else []
        return {"answer": n, "unit": "count", "confidence": _level(conf), "confidence_score": conf,
                "parsed_by": "cache" if rec.get("cached") else "llm", "answered_by": "hybrid" if agree else "vlm",
                "evidence": ev, "cost_usd": rec["cost_usd"], "call_cost_shared": True, "route": route,
                "latency_ms": int((time.time() - t0) * 1000) if rec.get("cached") else rec["latency_ms"]}
    raise ValueError(route)
