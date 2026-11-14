"""Door experiment on drawing sheets with hand-labelled ground truth (eval/doors_gt.json).

Variants:
  M0  product as is (doors in data/scenes/<plan>/scene.json, mapped to original pixels)
  M1  candidates: YOLO on the whole sheet (product scale, low threshold) + ALL door arcs (no wall filter)
  M2  M1 + Opus verifies crops around each candidate (one call per batch of crops)
  M3  candidates: YOLO on tiles at TARGET_PPM (core 1024 px + 1.5 m margin, centre ownership) + all arcs
  M4  M3 + Opus verification

    .venv/bin/python scripts/exp_doors.py --sheets-dir ../plans --scenes data/scenes [--no-llm]
Matching: predicted centre within TOL_M of a labelled door (greedy, one-to-one).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Image.MAX_IMAGE_PIXELS = None
for _l in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    if "=" in _l and not _l.lstrip().startswith("#"):
        _k, _v = _l.split("=", 1)
        if _v.strip():
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

TOL_M = 1.0
TARGET_PPM = 55.0
CORE = 1024
MARGIN_M = 1.5
CAND_CONF = 0.15          # YOLO door threshold for candidates (product: 0.35)
DEDUP_M = 0.8             # YOLO box and arc closer than this = same candidate
CROP_M = 3.0              # verification crop side (m)
CROP_PX = 320             # crop resized to this for the LLM
BATCH = 16
MODEL = os.environ.get("EXP_MODEL", "claude-opus-5-5")
OUT = ROOT / "out" / "exp_doors"
SCENE_DIRS = {"M-101A": "TEST-1-cc39304e7d", "S-100F": "TEST-2-508c207e8d", "school-fragment": "TEST-4-20615c86be"}
spent = {"usd": 0.0}


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def in_region(p, region):
    return region is None or (region[0] <= p[0] <= region[2] and region[1] <= p[1] <= region[3])


# ---------------- M0: product ----------------
def product_doors(scene_dir: Path) -> list[dict]:
    s = json.loads((scene_dir / "scene.json").read_text())
    m = s["meta"]
    crop = m.get("crop") or [0, 0, *m["orig_size"]]
    f = s["width"] / (crop[2] - crop[0])
    out = []
    for o in s["objects"]:
        if o.get("class") != "door":
            continue
        p = np.array(o["polygon"]) / f + np.array(crop[:2])
        out.append({"c": p.mean(0).tolist(), "box": [*p.min(0), *p.max(0)], "src": o.get("source"), "conf": o["conf"]})
    return out


# ---------------- candidates ----------------
def yolo_doors(bgr: np.ndarray, conf: float) -> list[tuple[np.ndarray, float]]:
    from fp.infer import THRESHOLDS, detect_yolo
    th = json.loads(json.dumps(THRESHOLDS))
    th["door"]["conf"] = conf
    th["door"]["merge"] = th["door"].get("merge") or 0.1
    return [(np.asarray(p), c) for cls, p, c in detect_yolo(bgr, os.environ.get("FP_MODEL", str(ROOT / "runs/colab/v2_best.pt")),
                                                          thresholds=th) if cls == "door"]


def yolo_whole(gray, conf):
    """Like the product: whole sheet at <= 2048 px."""
    s = min(1.0, 2048 / max(gray.shape))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    out = []
    for p, c in yolo_doors(to_bgr(g), conf):
        p = p / s
        out.append({"c": p.mean(0).tolist(), "box": [*p.min(0), *p.max(0)], "src": "yolo", "conf": c})
    return out


def yolo_tiles(gray, ppm, conf):
    """Rescale to TARGET_PPM, tiles = core CORE px + margin; keep a detection only in the tile whose core holds its centre."""
    s = TARGET_PPM / ppm
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    H, W = g.shape
    m = int(round(MARGIN_M * TARGET_PPM))
    out = []
    for y0 in range(0, H, CORE):
        for x0 in range(0, W, CORE):
            cx1, cy1 = min(W, x0 + CORE), min(H, y0 + CORE)
            tx0, ty0, tx1, ty1 = max(0, x0 - m), max(0, y0 - m), min(W, cx1 + m), min(H, cy1 + m)
            tile = g[ty0:ty1, tx0:tx1]
            if (tile < 200).mean() < 0.002:  # empty paper
                continue
            for p, c in yolo_doors(to_bgr(tile), conf):
                p = p + np.array([tx0, ty0])
                cen = p.mean(0)
                if x0 <= cen[0] < cx1 and y0 <= cen[1] < cy1:  # ownership
                    p = p / s
                    out.append({"c": (cen / s).tolist(), "box": [*p.min(0), *p.max(0)], "src": "yolo_tile", "conf": c})
    return out


def arcs_all(gray, ppm):
    from fp.doors_cv import find_door_arcs
    s = min(1.0, 80.0 / ppm)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    out = []
    for a in find_door_arcs(g, ppm * s, None, None):
        b = np.array(a["bbox"]) / s
        out.append({"c": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], "box": b.tolist(), "src": "arc", "conf": a["score"] / 2})
    return out


def merge(cands, ppm):
    keep = []
    for c in sorted(cands, key=lambda d: -d["conf"]):
        if all(np.hypot(c["c"][0] - k["c"][0], c["c"][1] - k["c"][1]) > DEDUP_M * ppm for k in keep):
            keep.append(dict(c))
        else:
            k = min(keep, key=lambda k: np.hypot(c["c"][0] - k["c"][0], c["c"][1] - k["c"][1]))
            k["src"] = "+".join(sorted(set(k["src"].split("+")) | {c["src"]}))
    return keep


# ---------------- verification ----------------
SCHEMA = {"type": "object", "properties": {"crops": {"type": "array", "items": {"type": "object", "properties": {
    "crop": {"type": "integer"}, "door": {"type": "boolean"}, "leaves": {"type": "integer"}},
    "required": ["crop", "door", "leaves"], "additionalProperties": False}}},
    "required": ["crops"], "additionalProperties": False}
from fp.door_verify import PROMPT  # noqa: E402 — the product prompt


def crop_png(gray, cand, ppm):
    half = CROP_M * ppm / 2
    cx, cy = cand["c"]
    x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    H, W = gray.shape
    pad = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    im = cv2.cvtColor(cv2.resize(pad, (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    k = CROP_PX / (x1 - x0)
    b = cand["box"]
    cv2.rectangle(im, (int((b[0] - x0) * k) - 3, int((b[1] - y0) * k) - 3), (int((b[2] - x0) * k) + 3,
                  int((b[3] - y0) * k) + 3), (0, 0, 255), 2)
    ok, buf = cv2.imencode(".png", im)
    return buf.tobytes()


def verify(gray, cands, ppm, tag):
    import anthropic
    cdir = OUT / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    for i in range(0, len(cands), BATCH):
        part = cands[i:i + BATCH]
        imgs = [crop_png(gray, c, ppm) for c in part]
        key = hashlib.sha256(b"".join(imgs) + PROMPT.encode() + MODEL.encode()).hexdigest()
        f = cdir / f"{key}.json"
        if f.exists():
            rec = json.loads(f.read_text())
        else:
            content = []
            for n, b in enumerate(imgs, 1):
                content += [{"type": "text", "text": f"Crop {n}:"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                         "data": base64.standard_b64encode(b).decode()}}]
            content.append({"type": "text", "text": PROMPT})
            r = client.messages.create(model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": content}],
                                       output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
            data = json.loads(next(b.text for b in r.content if b.type == "text"))
            cost = (r.usage.input_tokens * 4 + r.usage.output_tokens * 20) / 1e6
            rec = {"data": data, "cost": cost}
            f.write_text(json.dumps(rec))
            spent["usd"] += cost
        ans = {c["crop"]: c for c in rec["data"]["crops"]}
        for n, c in enumerate(part, 1):
            a = ans.get(n, {"door": False, "leaves": 0})
            c["door"], c["leaves"] = bool(a["door"]), int(a["leaves"])
    return [c for c in cands if c["door"]]


# ---------------- scoring ----------------
def score(pred, gt, ppm, region):
    pred = [p for p in pred if in_region(p["c"], region)]
    gt = [g for g in gt if in_region(g, region)]
    used, tp = set(), 0
    for p in sorted(pred, key=lambda d: -d.get("conf", 0)):
        best = None
        for j, g in enumerate(gt):
            d = np.hypot(p["c"][0] - g[0], p["c"][1] - g[1])
            if j not in used and d <= TOL_M * ppm and (best is None or d < best[1]):
                best = (j, d)
        if best:
            used.add(best[0])
            tp += 1
    return {"pred": len(pred), "gt": len(gt), "tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets-dir", required=True)
    ap.add_argument("--scenes", default=str(ROOT / "data" / "scenes"))
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gtall = json.loads((ROOT / "eval" / "doors_gt.json").read_text())["sheets"]
    rows = []
    for name, g in gtall.items():
        t0 = time.time()
        gray = load_gray(Path(a.sheets_dir) / g["image"])
        ppm, region = g["px_per_m"], g["region"]
        work, off = gray, np.array([0, 0])
        if region:  # experiment speed: process the labelled region + 3 m context only
            m = int(3 * ppm)
            x0, y0 = max(0, region[0] - m), max(0, region[1] - m)
            work, off = gray[y0:region[3] + m, x0:region[2] + m], np.array([x0, y0])

        def shift(cs):
            for c in cs:
                c["c"] = (np.array(c["c"]) + off).tolist()
                c["box"] = (np.array(c["box"]).reshape(2, 2) + off).ravel().tolist()
            return cs
        res = {"M0": product_doors(Path(a.scenes) / SCENE_DIRS[name])}
        arcs = shift(arcs_all(work, ppm))
        res["M1"] = merge(shift(yolo_whole(gray, CAND_CONF)) + [dict(c) for c in arcs], ppm)
        res["M3"] = merge(shift(yolo_tiles(work, ppm, CAND_CONF)) + [dict(c) for c in arcs], ppm)
        if not a.no_llm:
            for src, dst in (("M1", "M2"), ("M3", "M4")):
                cand = [dict(c) for c in res[src] if in_region(c["c"], region)]
                res[dst] = verify(gray, cand, ppm, f"{name}-{dst}")
        for k, v in res.items():
            sc = score(v, g["doors"], ppm, region)
            rows.append({"sheet": name, "variant": k, **sc})
            print(f"{name:16s} {k}: pred {sc['pred']:3d}  gt {sc['gt']:3d}  TP {sc['tp']:3d}  FP {sc['fp']:3d}  FN {sc['fn']:3d}"
                  f"  count error {sc['pred'] - sc['gt']:+d}", flush=True)
        (OUT / f"{name}.json").write_text(json.dumps({k: v for k, v in res.items()}, default=float))
        print(f"  ({time.time() - t0:.0f}s, LLM so far ${spent['usd']:.3f})", flush=True)
    tot = {}
    for r in rows:
        t = tot.setdefault(r["variant"], {"tp": 0, "fp": 0, "fn": 0, "abs_err": 0})
        for k in ("tp", "fp", "fn"):
            t[k] += r[k]
        t["abs_err"] += abs(r["pred"] - r["gt"])
    print("\nTOTAL over 3 sheets (32 labelled doors):")
    for k, t in tot.items():
        p = t["tp"] / max(t["tp"] + t["fp"], 1)
        rc = t["tp"] / max(t["tp"] + t["fn"], 1)
        print(f"  {k}: TP {t['tp']:3d} FP {t['fp']:3d} FN {t['fn']:3d}  precision {p:.2f} recall {rc:.2f}"
              f"  sum |count error| {t['abs_err']}")
    print(f"LLM cost ${spent['usd']:.3f}")
    (OUT / "summary.json").write_text(json.dumps({"rows": rows, "total": tot, "llm_usd": spent["usd"]}))


if __name__ == "__main__":
    main()
