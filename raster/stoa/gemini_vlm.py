"""Gemini baseline: image + per-type definition -> structured JSON with count and boxes.

Two modes:
  single - one call per (image, question type)
  batch  - one call per image answering all question types (≈10x cheaper)
Responses are cached on disk by (sha256(image), sha256(prompt+schema), model, thinking budget).
"""
import hashlib
import json
import os
import time
from pathlib import Path

from . import config
from .qtypes import QTYPES

GLOSSARY = """Floor plans are mostly Finnish, sometimes Swedish. Room label abbreviations:
MH, SR, SOV = bedroom; OH, ALLRUM, VARDAGSRUM = living room; K, KK, KEITTIÖ, KÖK, TK, AVOK = kitchen (KK/keittokomero = kitchenette);
KH, KPH, PH, PESUH, SH, BADRUM, TVÄTT = bathroom/washroom; WC, TOA = toilet room; S, SAUNA, BASTU = sauna;
VH = walk-in closet (NOT a bedroom); ET, ETEINEN, HALL, TK(tuulikaappi) = entrance; PARVEKE, P, BALKONG, LASITETTU PARVEKE = balcony;
TERASSI/TERRASS = terrace (not a balcony); VAR, VARASTO, FÖRRÅD = storage; TEKN = technical room; KHH = utility room; AUTOK = garage;
"1. KERROS", "2. KERROS", "KELLARI", "ULLAKKO", "VÅNING" = floor/storey labels. Text may be rotated 90° or 180°."""

DEFS = {
    "windows": "windows drawn in the walls (each separate window opening counts as one)",
    "doors": "doors: doorways fitted with a door leaf (swing arc or sliding door symbol). Open doorways without a door leaf do NOT count",
    "toilets": "toilet bowls (WC seats)",
    "sinks": "sinks and wash basins, kitchen and bathroom combined; a double-bowl sink counts as ONE",
    "showers": "showers and bathtubs combined (each shower and each bathtub counts)",
    "saunas": "saunas (sauna rooms, e.g. labelled S, SAUNA, BASTU, or a room with sauna benches)",
    "kitchens": "kitchens, including kitchenettes (KK), open kitchens and utility kitchens",
    "bedrooms": "bedrooms (makuuhuone, usually labelled MH; Swedish SR/SOVRUM). VH (walk-in closet) is NOT a bedroom",
    "wc_rooms": "separate WC rooms (small toilet rooms labelled WC, separate from bathrooms)",
    "bathrooms": "bathrooms or washrooms (KH, KPH, PH, PESUH, SH), NOT counting separate WC rooms",
    "balconies": "balconies (parveke, including glazed balconies). Terraces are not balconies",
    "floors": "floors (storeys) shown on the image. Several separate plan drawings of different storeys on one image count separately",
}

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "box_2d": {"type": "array", "items": {"type": "integer"},
                   "description": "[ymin, xmin, ymax, xmax] normalized to 0-1000"},
    },
    "required": ["label", "box_2d"],
}
TYPE_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": ITEM_SCHEMA}, "count": {"type": "integer"}},
    "required": ["items", "count"],
    "propertyOrdering": ["items", "count"],
}


def prompt_single(qtype: str) -> str:
    return (f"You are an expert at reading architectural floor plans.\n{GLOSSARY}\n\n"
            f"Task: count the {DEFS[qtype]}.\n"
            "First list every instance in `items` with a short label and a bounding box, then give `count`. "
            "Look carefully at the whole image; do not count the same object twice.")


def prompt_batch() -> str:
    lines = "\n".join(f"- {q}: {DEFS[q]}" for q in QTYPES)
    return (f"You are an expert at reading architectural floor plans.\n{GLOSSARY}\n\n"
            "For each key below, list every instance in `items` (short label + bounding box), then give `count`.\n"
            f"{lines}\n"
            "For room-type keys, the box is the room. For floors, the box is each storey drawing. "
            "Look carefully at the whole image; do not count the same object twice.")


def schema_batch():
    return {"type": "object", "properties": {q: TYPE_SCHEMA for q in QTYPES},
            "required": QTYPES, "propertyOrdering": QTYPES}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def price(model: str, usage: dict) -> float:
    pin, pout = config.PRICES.get(model, (float("nan"), float("nan")))
    return (usage.get("in", 0) * pin + usage.get("out", 0) * pout) / 1e6


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def call(image_path: Path, prompt: str, schema: dict, model: str | None = None,
         image_bytes: bytes | None = None, mime="image/png") -> dict:
    """Returns {"data": parsed_json, "usage": {...}, "cost_usd", "cached": bool, "latency_ms"}."""
    model = model or config.GEMINI_MODEL
    img = image_bytes if image_bytes is not None else Path(image_path).read_bytes()
    key = _sha((_sha(img) + _sha((prompt + json.dumps(schema, sort_keys=True)).encode()) + model
                + str(config.GEMINI_THINKING_BUDGET) + config.GEMINI_MEDIA_RESOLUTION).encode())
    cdir = config.CACHE_DIR / "gemini"
    cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / f"{key}.json"
    if cfile.exists():
        rec = json.loads(cfile.read_text())
        rec["cached"] = True
        return rec

    from google.genai import types
    cfg = dict(response_mime_type="application/json", response_schema=schema, temperature=0)
    if config.GEMINI_THINKING_BUDGET >= 0:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=config.GEMINI_THINKING_BUDGET)
    if config.GEMINI_MEDIA_RESOLUTION:
        cfg["media_resolution"] = f"MEDIA_RESOLUTION_{config.GEMINI_MEDIA_RESOLUTION.upper()}"
    t0 = time.time()
    for attempt in range(5):
        try:
            resp = _get_client().models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=img, mime_type=mime), prompt],
                config=types.GenerateContentConfig(**cfg))
            data = json.loads(resp.text)
            break
        except Exception as e:  # rate limits / transient / malformed JSON
            if attempt == 4:
                raise
            time.sleep(2 ** attempt * 2)
            last = e
    um = resp.usage_metadata
    usage = {"in": um.prompt_token_count or 0,
             "out": (um.candidates_token_count or 0) + (um.thoughts_token_count or 0),
             "thoughts": um.thoughts_token_count or 0}
    rec = {"data": data, "usage": usage, "model": model, "cost_usd": price(model, usage),
           "latency_ms": int((time.time() - t0) * 1000)}
    cfile.write_text(json.dumps(rec))
    rec["cached"] = False
    return rec
