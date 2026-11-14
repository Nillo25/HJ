"""Claude VLM: image + per-type definitions -> structured JSON (counts + optional boxes).

Same prompts/definitions as gemini_vlm. Boxes are [x1, y1, x2, y2] normalized to 0-1000.
Cached on disk by (sha256(image), sha256(prompt+schema), model, effort).
"""
import base64
import hashlib
import json
import time
from pathlib import Path

import anthropic

from . import config
from .gemini_vlm import DEFS, GLOSSARY
from .qtypes import QTYPES

ITEM = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "box": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["label", "box"],
    "additionalProperties": False,
}


def type_schema(boxes: bool):
    if boxes:
        return {"type": "object", "properties": {"items": {"type": "array", "items": ITEM}, "count": {"type": "integer"}},
                "required": ["items", "count"], "additionalProperties": False}
    return {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"],
            "additionalProperties": False}


def schema_batch(qtypes=QTYPES, boxes=True):
    counts = {"type": "object", "properties": {q: {"type": "integer"} for q in qtypes},
              "required": list(qtypes), "additionalProperties": False}
    if not boxes:
        return {"type": "object", "properties": {"counts": counts}, "required": ["counts"], "additionalProperties": False}
    obj = {"type": "object", "properties": {"type": {"type": "string", "enum": list(qtypes)},
                                            "box": {"type": "array", "items": {"type": "integer"}}},
           "required": ["type", "box"], "additionalProperties": False}
    return {"type": "object", "properties": {"objects": {"type": "array", "items": obj}, "counts": counts},
            "required": ["objects", "counts"], "additionalProperties": False}


def prompt_batch(qtypes=QTYPES, boxes=True) -> str:
    lines = "\n".join(f"- {q}: {DEFS[q]}" for q in qtypes)
    how = ("First list every counted instance in `objects` (its type key + box [x1, y1, x2, y2] normalized to 0-1000 "
           "of image width/height), then fill `counts` for every key.\n" if boxes
           else "Fill `counts` for every key.\n")
    return (f"You are an expert at reading architectural floor plans.\n{GLOSSARY}\n\nCount these things:\n{lines}\n\n{how}"
            "For room-type keys, the box is the room. For floors, the box is each storey drawing. "
            "Look carefully at the whole image; do not count the same object twice.")


MAX_BYTES = 9_000_000  # API limit is 10 MB per image


def fit_image(img: bytes, mime: str):
    """Downscale/re-encode images over the API size limit; small images pass through unchanged."""
    if len(img) <= MAX_BYTES:
        return img, mime
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(img)).convert("RGB")
    while True:
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        if buf.tell() <= MAX_BYTES:
            return buf.getvalue(), "image/png"
        im = im.resize((int(im.width * 0.8), int(im.height * 0.8)), Image.LANCZOS)


def price(model, usage):
    pin, pout = config.CLAUDE_PRICES.get(model, (float("nan"), float("nan")))
    return (usage["in"] * pin + usage["out"] * pout) / 1e6


_client = None
BUDGET_FILE = config.CACHE_DIR / "budget.json"  # {"baseline_usd": spent before the cap, "cap_usd": extra allowed}


class BudgetExceeded(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def spent_total() -> float:
    return sum(json.loads(f.read_text()).get("cost_usd", 0) for f in (config.CACHE_DIR / "claude").glob("*.json"))


def check_budget(extra_estimate: float = 0.0):
    """Raise if spend since the baseline (+ an estimate for what we are about to send) exceeds the cap."""
    if not BUDGET_FILE.exists():
        return
    b = json.loads(BUDGET_FILE.read_text())
    used = spent_total() - b["baseline_usd"]
    if used + extra_estimate > b["cap_usd"]:
        raise BudgetExceeded(f"budget: used ${used:.3f} + planned ${extra_estimate:.3f} > cap ${b['cap_usd']:.2f}")


def _client_get():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def _prepare(image_path, prompt, schema, model, effort, image_bytes, mime):
    model = model or config.CLAUDE_MODEL
    effort = effort or config.CLAUDE_EFFORT
    img = image_bytes if image_bytes is not None else Path(image_path).read_bytes()
    img, mime = fit_image(img, mime)
    key = _sha((_sha(img) + _sha((prompt + json.dumps(schema, sort_keys=True)).encode()) + model + effort).encode())
    params = dict(
        model=model,
        max_tokens=16000,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime,
                                         "data": base64.standard_b64encode(img).decode()}},
            {"type": "text", "text": prompt},
        ]}],
    )
    cdir = config.CACHE_DIR / "claude"
    cdir.mkdir(parents=True, exist_ok=True)
    return key, cdir / f"{key}.json", params


def _record(msg, model, effort, t0, discount=1.0):
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {msg.stop_details}")
    text = next(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    usage = {"in": (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
             "out": u.output_tokens or 0}
    return {"data": json.loads(text), "usage": usage, "model": model, "effort": effort,
            "cost_usd": price(model, usage) * discount, "list_cost_usd": price(model, usage),
            "batch": discount != 1.0, "latency_ms": int((time.time() - t0) * 1000)}


def call(image_path: Path, prompt: str, schema: dict, model: str | None = None, effort: str | None = None,
         image_bytes: bytes | None = None, mime="image/png") -> dict:
    key, cfile, params = _prepare(image_path, prompt, schema, model, effort, image_bytes, mime)
    if cfile.exists():
        rec = json.loads(cfile.read_text())
        rec["cached"] = True
        return rec
    check_budget()
    t0 = time.time()
    resp = _client_get().messages.create(**params)
    rec = _record(resp, params["model"], params["output_config"]["effort"], t0)
    cfile.write_text(json.dumps(rec))
    rec["cached"] = False
    return rec


def call_batch(jobs: list[dict], est_cost_each: float, poll_s: int = 30, log=print) -> int:
    """Run uncached jobs through the Message Batches API (50% price) and fill the same cache.
    jobs: dicts with keys of call(): image_path, prompt, schema, model, effort, image_bytes (optional).
    Returns number of requests sent. Afterwards call() for the same job returns the cached result."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    todo = {}
    for j in jobs:
        key, cfile, params = _prepare(j.get("image_path"), j["prompt"], j["schema"], j.get("model"), j.get("effort"),
                                      j.get("image_bytes"), j.get("mime", "image/png"))
        if not cfile.exists():
            todo[key] = (cfile, params)
    if not todo:
        return 0
    check_budget(est_cost_each * 0.5 * len(todo))
    t0 = time.time()
    batch = _client_get().messages.batches.create(requests=[
        Request(custom_id=k, params=MessageCreateParamsNonStreaming(**p)) for k, (_, p) in todo.items()])
    log(f"batch {batch.id}: {len(todo)} requests")
    while True:
        b = _client_get().messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        log(f"  {b.processing_status}: processing {b.request_counts.processing}, done {b.request_counts.succeeded}"
            f" ({time.time() - t0:.0f}s)")
        time.sleep(poll_s)
    ok = 0
    for r in _client_get().messages.batches.results(batch.id):
        cfile, params = todo[r.custom_id]
        if r.result.type == "succeeded":
            try:
                rec = _record(r.result.message, params["model"], params["output_config"]["effort"], t0, discount=0.5)
                cfile.write_text(json.dumps(rec))
                ok += 1
            except Exception as e:  # noqa: BLE001
                log(f"  {r.custom_id[:12]}: bad result {e}")
        else:
            log(f"  {r.custom_id[:12]}: {r.result.type}")
    log(f"batch done: {ok}/{len(todo)} ok, {time.time() - t0:.0f}s")
    return len(todo)
