"""Claude verification with CubiCasa hints: draw numbered candidate boxes, ask which are wrong / what's missing."""
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import claude_vlm, config, cubicasa_count
from .gemini_vlm import DEFS, GLOSSARY

COLORS = {"windows": (0, 90, 255), "doors": (230, 0, 0)}  # RGB
TARGET_LONG = 1500


def render(path: Path, cands: dict) -> tuple[bytes, dict]:
    """cands: {qtype: [objects]}; returns PNG bytes and {qtype: {num: box}}."""
    im = np.array(Image.open(path).convert("RGB"))
    s = TARGET_LONG / max(im.shape[:2])
    if s > 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    else:
        s = 1.0
    numbering = {}
    n = 0
    fs = 0.5
    for qt, objs in cands.items():
        col = COLORS[qt]
        numbering[qt] = {}
        for o in objs:
            n += 1
            x1, y1, x2, y2 = [int(v * s) for v in o["box"]]
            cv2.rectangle(im, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), col, 2)
            label = str(n)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            tx, ty = x2 + 3, max(y1 + th, th + 2)
            if tx + tw > im.shape[1]:
                tx = max(0, x1 - tw - 5)
            cv2.rectangle(im, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2), (255, 255, 255), -1)
            cv2.putText(im, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
            numbering[qt][n] = o["box"]
    buf = io.BytesIO()
    Image.fromarray(im).save(buf, format="PNG")
    return buf.getvalue(), numbering


def prompt(numbering: dict) -> str:
    parts = []
    for qt, nums in numbering.items():
        color = "blue" if qt == "windows" else "red"
        ids = ", ".join(map(str, nums)) or "none"
        parts.append(f"- {qt} ({DEFS[qt]}): a detector proposed {len(nums)} candidates, drawn as {color} "
                     f"numbered boxes: {ids}.")
    return (f"You are an expert at reading architectural floor plans.\n{GLOSSARY}\n\n"
            "An automatic detector marked candidate objects on this floor plan:\n" + "\n".join(parts) + "\n\n"
            "Check every candidate. For each type report: `false_positives` = numbers of boxes that are NOT a real "
            "instance (or duplicate another box for the same object); `merged` = numbers of boxes that cover MORE "
            "than one real instance, with how many instances each actually covers; `missed` = real instances that "
            "have no box at all; then `final_count`. Windows in a bay or a row separated by wall pieces are separate "
            "windows. Be precise: the exact count matters.")


def schema(qtypes):
    merged = {"type": "object", "properties": {"box": {"type": "integer"}, "instances": {"type": "integer"}},
              "required": ["box", "instances"], "additionalProperties": False}
    per = {"type": "object", "properties": {
        "false_positives": {"type": "array", "items": {"type": "integer"}},
        "merged": {"type": "array", "items": merged},
        "missed": {"type": "integer"},
        "final_count": {"type": "integer"}},
        "required": ["false_positives", "merged", "missed", "final_count"], "additionalProperties": False}
    return {"type": "object", "properties": {q: per for q in qtypes}, "required": list(qtypes),
            "additionalProperties": False}


def verify(path: Path, qtypes=("windows", "doors"), model=None, effort=None) -> dict:
    stem = Path(path).stem
    cands = {qt: cubicasa_count.objects(stem, qt) for qt in qtypes}
    img, numbering = render(path, cands)
    rec = claude_vlm.call(path, prompt(numbering), schema(list(qtypes)), model=model, effort=effort,
                          image_bytes=img)
    out = {}
    for qt in qtypes:
        r = rec["data"][qt]
        fp = set(r["false_positives"]) & set(numbering[qt])
        extra = sum(max(0, m["instances"] - 1) for m in r["merged"] if m["box"] in numbering[qt])
        derived = len(numbering[qt]) - len(fp) + extra + r["missed"]
        out[qt] = {"cubicasa": len(numbering[qt]), "final_count": r["final_count"], "derived": derived,
                   "kept_boxes": [b for k, b in numbering[qt].items() if k not in fp], "raw": r}
    return {"result": out, "cost_usd": rec["cost_usd"], "cached": rec.get("cached", False), "image": img}


def prompt_panes(numbering: dict) -> str:
    nums = numbering["windows"]
    ids = ", ".join(map(str, nums)) or "none"
    return (f"You are an expert at reading architectural floor plans.\n{GLOSSARY}\n\n"
            f"A detector marked {len(nums)} candidate windows as blue numbered boxes: {ids}.\n"
            "COUNTING RULE: every window section (pane/sash) separated by a mullion (a short line across the "
            "window frame, perpendicular to the wall) counts as a SEPARATE window. A window drawn with one mullion "
            "in the middle counts as 2, with two mullions as 3, etc.\n"
            "For each box give `panes` (0 if the box is not a window at all). Then report `missed_panes`: window "
            "panes in walls that have no box at all (count them by the same rule). Then `final_count` = sum of "
            "panes + missed_panes. Zoom in mentally on each box; the exact count matters.")


def schema_panes():
    box = {"type": "object", "properties": {"box": {"type": "integer"}, "panes": {"type": "integer"}},
           "required": ["box", "panes"], "additionalProperties": False}
    return {"type": "object", "properties": {"boxes": {"type": "array", "items": box},
                                             "missed_panes": {"type": "integer"}, "final_count": {"type": "integer"}},
            "required": ["boxes", "missed_panes", "final_count"], "additionalProperties": False}


def verify_windows(path: Path, model=None, effort=None) -> dict:
    stem = Path(path).stem
    cands = {"windows": cubicasa_count.objects(stem, "windows")}
    img, numbering = render(path, cands)
    rec = claude_vlm.call(path, prompt_panes(numbering), schema_panes(), model=model, effort=effort, image_bytes=img)
    d = rec["data"]
    per = {b["box"]: b["panes"] for b in d["boxes"] if b["box"] in numbering["windows"]}
    derived = sum(per.get(k, 1) for k in numbering["windows"]) + d["missed_panes"]
    kept = [{"box": numbering["windows"][k], "panes": per.get(k, 1)} for k in numbering["windows"] if per.get(k, 1) > 0]
    return {"final_count": d["final_count"], "derived": derived, "cubicasa": len(numbering["windows"]),
            "kept": kept, "cost_usd": rec["cost_usd"], "cached": rec.get("cached", False)}
