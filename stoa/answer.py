"""Module entry point: answer(path, question, method) -> team answer contract dict."""
import time
from pathlib import Path

from PIL import Image

from . import config, gemini_vlm
from .qtypes import qtype_of

VLM_CONF = 0.6


def _level(c: float) -> str:
    return "high" if c >= 0.9 else "medium" if c >= 0.7 else "low"


def _boxes_from_gemini(items, w, h, label):
    ev = []
    for it in items or []:
        b = it.get("box_2d") or []
        if len(b) != 4:
            continue
        y1, x1, y2, x2 = b
        ev.append({"kind": "bbox", "page": 1,
                   "box": [round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)],
                   "label": it.get("label") or label, "score": VLM_CONF})
    return ev


def answer_gemini(path: Path, qtype: str, mode: str = "batch", model: str | None = None) -> dict:
    t0 = time.time()
    if mode == "batch":
        rec = gemini_vlm.call(path, gemini_vlm.prompt_batch(), gemini_vlm.schema_batch(), model)
        part = rec["data"].get(qtype, {})
    else:
        rec = gemini_vlm.call(path, gemini_vlm.prompt_single(qtype), gemini_vlm.TYPE_SCHEMA, model)
        part = rec["data"]
    w, h = Image.open(path).size
    count = int(part.get("count", 0) or 0)
    conf = VLM_CONF if count > 0 else min(VLM_CONF, 0.4)
    return {
        "answer": count, "unit": "count", "confidence": _level(conf), "confidence_score": conf,
        "parsed_by": "cache" if rec.get("cached") else "llm", "answered_by": "vlm",
        "evidence": _boxes_from_gemini(part.get("items"), w, h, qtype),
        "n_items": len(part.get("items") or []),
        # cost of the underlying call; in batch mode shared across all question types of this image
        "cost_usd": rec["cost_usd"], "call_cost_shared": mode == "batch",
        "latency_ms": int((time.time() - t0) * 1000) if rec.get("cached") else rec["latency_ms"],
    }


def answer_claude(path: Path, qtype: str, model: str | None = None) -> dict:
    from . import claude_vlm
    t0 = time.time()
    boxes = config.CLAUDE_BOXES
    rec = claude_vlm.call(path, claude_vlm.prompt_batch(boxes=boxes), claude_vlm.schema_batch(boxes=boxes), model)
    count = int(rec["data"]["counts"].get(qtype, 0) or 0)
    w, h = Image.open(path).size
    ev = []
    for o in rec["data"].get("objects", []):
        if o.get("type") == qtype and len(o.get("box") or []) == 4:
            x1, y1, x2, y2 = o["box"]
            ev.append({"kind": "bbox", "page": 1, "box": [round(x1 * w / 1000), round(y1 * h / 1000),
                       round(x2 * w / 1000), round(y2 * h / 1000)], "label": qtype, "score": VLM_CONF})
    conf = VLM_CONF if count > 0 else min(VLM_CONF, 0.4)
    return {"answer": count, "unit": "count", "confidence": _level(conf), "confidence_score": conf,
            "parsed_by": "cache" if rec.get("cached") else "llm", "answered_by": "vlm", "evidence": ev,
            "cost_usd": rec["cost_usd"], "call_cost_shared": True,
            "latency_ms": int((time.time() - t0) * 1000) if rec.get("cached") else rec["latency_ms"]}


def answer(path: str | Path, question: str, method: str = "gemini_batch", **kw) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = config.DATASET_DIR / path
    qtype = qtype_of(question)
    if method == "gemini_batch":
        return answer_gemini(path, qtype, "batch", **kw)
    if method == "gemini_single":
        return answer_gemini(path, qtype, "single", **kw)
    if method == "claude":
        return answer_claude(path, qtype, **kw)
    if method == "hybrid":
        from . import router
        return router.answer(path, qtype, **kw)
    if method == "cubicasa":
        from . import cubicasa_count
        return cubicasa_count.answer(path, qtype)
    raise ValueError(method)
