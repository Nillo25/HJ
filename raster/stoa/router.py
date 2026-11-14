"""Hybrid router: per question type, use the method that won on the bench.

ROUTE values:
  "opus"        - Claude (CLAUDE_MODEL) one batch call per image, counts only; CubiCasa boxes as evidence if counts agree
  "verify"      - CubiCasa candidates drawn as numbered boxes, Claude marks false positives / misses
  "cubicasa"    - CubiCasa masks only ($0)
Filled from bench results (reports/bench_*.csv, reports/verify_results.json), see README of the run.
"""
import time
from pathlib import Path

from PIL import Image

from . import claude_vlm, config, cubicasa_count, verify  # noqa: F401

import os

PROFILES = {
    # 80 plans, all / test: 80.0% / 76.5%, ~$0.035 per plan
    "quality": {"windows": "cubicasa", "doors": "verify"},
    # scheme E: colleague YOLO for windows/doors, one Opus call per plan: 81.2% / 77.9%, ~$0.0145 per plan
    "cheap": {"windows": "hj", "doors": "hj"},
}
ROUTE = {
    "windows": "cubicasa",   # overridden by the profile below
    "doors": "verify",
    "toilets": "opus", "sinks": "opus", "showers": "opus", "saunas": "opus", "kitchens": "opus",
    "bedrooms": "opus", "wc_rooms": "opus", "bathrooms": "opus", "balconies": "opus", "floors": "opus",
}
ROUTE.update(PROFILES[os.environ.get("ROUTER_PROFILE", "cheap")])
HJ_LEVEL = {"high": 0.95, "medium": 0.75, "low": 0.5}
CUBICASA_HAS = {"windows", "doors", "toilets", "sinks", "saunas", "kitchens", "bedrooms", "wc_rooms",
                "bathrooms", "balconies", "floors"}


def _level(c):
    return "high" if c >= 0.9 else "medium" if c >= 0.7 else "low"


def _ev(objs):
    return [{"kind": "bbox", "page": 1, "box": o["box"], "label": o["label"],
             "score": o["score"]} for o in objs]


# Plan-level reliability ("брак"): independent sources (Claude, CubiCasa, colleague YOLO) disagree on many question
# types -> the plan is hard, answers on it are unreliable. On 80 plans, flagging the worst ~20% by this score:
# threshold 0.40 flags ~30% of plans: 68.7% accurate vs 85.8% for the rest (test: 71.4% vs 85.7%), reports/plan_reliability.txt.
UNRELIABLE_DISAGREEMENT = float(os.environ.get("STOA_UNRELIABLE_DISAGREEMENT", "0.40"))
_plan_cache: dict = {}


def cubicasa_ready(path: Path) -> bool:
    """CubiCasa masks for this image exist or can be computed now (weights installed). Optional dependency."""
    path = Path(path)
    out = config.CACHE_DIR / "cubicasa"
    if (out / f"{path.stem}.npz").exists():
        return True
    if not config.CUBICASA_WEIGHTS.exists():
        return False
    try:
        from . import cubicasa_infer
        out.mkdir(parents=True, exist_ok=True)
        cubicasa_infer.process_auto(path, out)
        return True
    except ImportError as e:  # CubiCasa deps not installed (stoa/setup_cubicasa.sh) -> work without its votes
        print(f"stoa: CubiCasa unavailable ({e}); continuing with Claude + HJ YOLO only")
        return False


def plan_check(path: Path, model: str | None = None) -> dict:
    """Disagreement of other sources with our answers over all 12 question types (0 = all agree, 1 = none)."""
    from .qtypes import QTYPES
    path = Path(path)
    key = (str(path), model)
    if key in _plan_cache:
        return _plan_cache[key]
    from . import hj_yolo
    hj = hj_yolo.detect(path)["counts"]
    per, disagree = [], []
    for qt in QTYPES:
        a = _answer_raw(path, qt, model)["answer"]
        votes = []
        if qt in CUBICASA_HAS and cubicasa_ready(path):
            votes.append(cubicasa_count.answer(path, qt)["answer"])
        if qt in hj:
            votes.append(hj[qt])
        if not votes:
            continue
        d = sum(v != a for v in votes) / len(votes)
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
    stem = path.stem
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
    cc = cubicasa_count.answer(path, qtype) if qtype in CUBICASA_HAS and cubicasa_ready(path) else None
    if route == "cubicasa":
        r = dict(cc)
        r["route"] = route
        return r
    if route == "opus":
        rec = claude_vlm.call(path, claude_vlm.prompt_batch(boxes=False), claude_vlm.schema_batch(boxes=False), model)
        n = int(rec["data"]["counts"].get(qtype, 0) or 0)
        agree = cc is not None and cc["answer"] == n
        conf = 0.95 if agree else (0.6 if n > 0 else 0.4)
        ev = cc["evidence"] if agree else []
        return {"answer": n, "unit": "count", "confidence": _level(conf), "confidence_score": conf,
                "parsed_by": "cache" if rec.get("cached") else "llm", "answered_by": "hybrid" if agree else "vlm",
                "evidence": ev, "cost_usd": rec["cost_usd"], "call_cost_shared": True, "route": route,
                "latency_ms": int((time.time() - t0) * 1000) if rec.get("cached") else rec["latency_ms"]}
    if route == "verify":
        v = verify.verify(path, ("windows", "doors"), model=model)
        res = v["result"][qtype]
        n = int(res["final_count"])
        conf = 0.95 if n == res["cubicasa"] else 0.6
        ev = [{"kind": "bbox", "page": 1, "box": b, "label": qtype.rstrip("s"), "score": 0.9}
              for b in res["kept_boxes"]]
        return {"answer": n, "unit": "count", "confidence": _level(conf), "confidence_score": conf,
                "parsed_by": "cache" if v["cached"] else "llm", "answered_by": "hybrid", "evidence": ev,
                "cost_usd": v["cost_usd"], "call_cost_shared": True, "route": route,
                "latency_ms": int((time.time() - t0) * 1000)}
    raise ValueError(route)
