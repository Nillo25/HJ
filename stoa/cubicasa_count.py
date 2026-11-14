"""Counting from cached CubiCasa masks. Every qtype -> list of objects (box, score)."""
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import config

# room classes
BG, OUTDOOR, WALL, KITCHEN, LIVING, BEDROOM, BATH, ENTRY, RAILING, STORAGE, GARAGE, UNDEF = range(12)
# icon classes
NOICON, WINDOW, DOOR, CLOSET, APPLIANCE, TOILET, SINK, SAUNA, FIREPLACE, BATHTUB, CHIMNEY = range(11)

# Tunable params; areas are fractions of the building area (non-bg, non-outdoor pixels) so they are scale-free.
P = {
    "icon_min_frac": 0.0004,      # min icon component area / building area
    "icon_min_px": 12,
    "room_min_frac": 0.008,       # min room component area / building area
    "room_min_px": 150,
    "wc_max_rel": 0.65,           # bath region with toilet & no tub & area < wc_max_rel * largest bath -> WC
    "floor_min_rel": 0.15,        # building block area / largest block
    "floor_dilate": 0.01,         # dilation kernel as fraction of image diagonal
    "sauna_merge": 0.01,          # merge sauna bench blobs within this fraction of diagonal
    "min_prob": 0.0,              # discard components whose mean prob below this
    "icon_min_m": 0.075,          # physical cap on the min icon side (m): area threshold <= (icon_min_m*px_per_m)^2
    "door_m": 0.85,
    "big_min_prob": 0.6,          # big sheets only: drop icon blobs with mean prob below this (chairs/cisterns on TEST-4;
                                  # calibrated on ONE sheet, 12/12 toilets kept, 3/3 false ones dropped)
    # doors are thin openings in walls: a building-relative area threshold dropped 60-90% of them on big plans.
    # Door-specific filter (tuned on in_test=False plans: 35% -> 46%, bias -1.97 -> -0.25):
    "door_dilate": 3,             # merge fragments of one opening (px at scale 1)
    "door_min_px": 12,            # absolute min area of door pixels (at scale 1, scaled by scale^2)
    "door_min_prob": 0.5,         # min mean prob of door pixels
    "big_m2": 500,               # building area (m^2) above which the physical icon threshold is used               # typical door leaf width, used to estimate px_per_m from detected doors
}


def estimate_px_per_m(icon_cls: np.ndarray, door_m: float | None = None) -> float | None:
    """px per metre from the median long side of detected door blobs (None if < 3 doors)."""
    n, _, st, _ = cv2.connectedComponentsWithStats((icon_cls == DOOR).astype(np.uint8), connectivity=4)
    sides = [max(st[i, 2], st[i, 3]) for i in range(1, n) if st[i, 4] >= 4]
    if len(sides) < 3:
        return None
    return float(np.median(sides)) / (door_m or P["door_m"])


import os

# which cached inference to use per question type, e.g. {"windows": "cubicasa_s2"}; default "cubicasa"
VARIANT = {}
DEFAULT_VARIANT = os.environ.get("CUBICASA_VARIANT", "cubicasa")


@lru_cache(maxsize=64)
def load(stem: str, variant: str = "cubicasa"):
    d = np.load(config.CACHE_DIR / variant / f"{stem}.npz")
    out = {k: d[k] for k in d.files}
    out.setdefault("scale", np.float32(1.0))
    out.setdefault("offset", np.array([0, 0], dtype=np.int32))  # crop offset in original pixels
    out.setdefault("big_sheet", np.bool_(False))  # set by cubicasa_infer --auto for large drawings
    if "px_per_m" not in out:
        out["px_per_m"] = np.float32(estimate_px_per_m(out["icon_cls"]) or 0)
    return out


def _components(mask, prob, min_area, label, min_prob=None):
    min_prob = P["min_prob"] if min_prob is None else min_prob
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        s = float(prob[lab == i].mean()) / 255.0
        if s < min_prob:
            continue
        out.append({"box": [int(x), int(y), int(x + w), int(y + h)], "score": round(s, 3), "label": label,
                    "area": int(a), "_id": i, "_lab": lab})
    return out


def _overlaps(mask_bool, box):
    x1, y1, x2, y2 = box
    return mask_bool[y1:y2, x1:x2]


def doors(ic, ip, sc=1.0):
    """door openings: dilate to merge fragments, then absolute area + mean-prob filter on the original door pixels."""
    m = (ic == DOOR).astype(np.uint8)
    k = max(1, int(round(P["door_dilate"] * sc)))
    mm = cv2.dilate(m, np.ones((k, k), np.uint8)) if k > 1 else m
    n, lab, st, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
    out = []
    for i in range(1, n):
        sel = (lab == i) & (m > 0)
        a = int(sel.sum())
        if a < P["door_min_px"] * sc * sc:
            continue
        s = float(ip[sel].mean()) / 255.0
        if s < P["door_min_prob"]:
            continue
        ys, xs = np.nonzero(sel)
        out.append({"box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "score": round(s, 3),
                    "label": "door", "area": a, "_id": i, "_lab": lab})
    return out


def objects(stem: str, qtype: str, variant: str | None = None):
    """Objects with "box" in ORIGINAL image pixels (scale/crop undone) and "mbox" in mask pixels."""
    variant = variant or VARIANT.get(qtype, DEFAULT_VARIANT)
    d = load(stem, variant)
    sc, (ox, oy) = float(d["scale"]), d["offset"]
    out = _objects_mask(stem, qtype, variant)
    for o in out:
        o["mbox"] = o["box"]
        x1, y1, x2, y2 = o["box"]
        o["box"] = [round(x1 / sc) + int(ox), round(y1 / sc) + int(oy), round(x2 / sc) + int(ox), round(y2 / sc) + int(oy)]
    return out


def _objects_mask(stem: str, qtype: str, variant: str):
    d = load(stem, variant)
    sc = float(d["scale"])
    rc, rp, ic, ip = d["room_cls"], d["room_prob"], d["icon_cls"], d["icon_prob"]
    H, W = rc.shape
    diag = float(np.hypot(H, W))
    bld_area = max(int(((rc != BG) & (rc != OUTDOOR)).sum()), 1)
    imin = P["icon_min_frac"] * bld_area
    ppm = float(d["px_per_m"])
    if bool(d["big_sheet"]) and ppm > 0:
        # big sheets: "building area" is huge, so the relative threshold would drop every icon -> physical cap
        imin = min(imin, (P["icon_min_m"] * ppm) ** 2)
    imin = max(P["icon_min_px"] * sc * sc, imin)
    rmin = max(P["room_min_px"] * sc * sc, P["room_min_frac"] * bld_area)

    big = bool(d["big_sheet"])

    def icons(cls, label):
        return _components(ic == cls, ip, imin, label, P["big_min_prob"] if big else None)

    def rooms(cls, label):
        return _components(rc == cls, rp, rmin, label)

    if qtype == "windows":
        return icons(WINDOW, "window")
    if qtype == "doors":
        if big:
            return icons(DOOR, "door")
        return doors(ic, ip, sc)
    if qtype == "toilets":
        return icons(TOILET, "toilet")
    if qtype == "sinks":
        return icons(SINK, "sink")
    if qtype == "showers":
        return icons(BATHTUB, "bathtub")  # no shower class: see hybrid / Gemini
    if qtype == "kitchens":
        return rooms(KITCHEN, "kitchen")
    if qtype == "bedrooms":
        return rooms(BEDROOM, "bedroom")
    if qtype in ("bathrooms", "wc_rooms"):
        baths = rooms(BATH, "bath")
        toilet = ic == TOILET
        tub = ic == BATHTUB
        sauna = ic == SAUNA
        big = max([b["area"] for b in baths], default=0)
        out = []
        for b in baths:
            m = b["_lab"] == b["_id"]
            # dilate a bit so icons touching the region boundary count
            m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            has_t, has_tub, has_s = (toilet & m).any(), (tub & m).any(), (sauna & m).any()
            is_wc = has_t and not has_tub and not has_s and b["area"] < P["wc_max_rel"] * big
            b["label"] = "wc" if is_wc else "bathroom"
            out.append(b)
        return [b for b in out if (b["label"] == "wc") == (qtype == "wc_rooms")]
    if qtype == "saunas":
        m = (ic == SAUNA).astype(np.uint8)
        k = max(1, int(P["sauna_merge"] * diag))
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
        return _components(m > 0, ip, imin * 2, "sauna")
    if qtype == "balconies":
        return rooms(RAILING, "balcony")
    if qtype == "floors":
        m = ((rc != BG) & (rc != OUTDOOR)).astype(np.uint8)
        k = max(1, int(P["floor_dilate"] * diag))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        comps = _components(m > 0, rp, 0, "floor")
        big = max([c["area"] for c in comps], default=0)
        return [c for c in comps if c["area"] >= P["floor_min_rel"] * big]
    raise ValueError(qtype)


def _level(c):
    return "high" if c >= 0.9 else "medium" if c >= 0.7 else "low"


def answer(path: Path, qtype: str) -> dict:
    t0 = time.time()
    stem = Path(path).stem
    objs = objects(stem, qtype)
    ev = [{"kind": "bbox", "page": 1, "box": o["box"], "label": o["label"], "score": o["score"]} for o in objs]
    conf = min([o["score"] for o in objs], default=0.4)
    if not objs:
        conf = min(conf, 0.4)
    return {"answer": len(objs), "unit": "count", "confidence": _level(conf), "confidence_score": conf,
            "parsed_by": "cache", "answered_by": "cubicasa", "evidence": ev, "cost_usd": 0.0,
            "latency_ms": int((time.time() - t0) * 1000)}
