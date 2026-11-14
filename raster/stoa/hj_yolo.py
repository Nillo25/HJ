"""Colleague's detector (repo HJ: YOLO-seg trained on CubiCasa5K, weights release weights-v2) as a source of
windows/doors (and other fixtures). HJ code is imported as a library, unchanged.

Per image: fp.infer.run(image, model, auto_scale=False) -> scene -> fp.query.count(scene, cls).
Cached to cache/hj/<stem>.json with counts, HJ's own calibrated confidence and boxes in ORIGINAL image pixels.
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from . import config

HJ_REPO = Path(os.environ.get("HJ_REPO", config.ROOT))  # stoa/ lives inside the HJ repo
_W = [HJ_REPO / "runs" / "colab" / n for n in ("floorplan_v2.pt", "v2_best.pt")]
HJ_MODEL = Path(os.environ.get("FP_MODEL") or next((str(w) for w in _W if w.exists()), str(_W[0])))
# our question type -> HJ classes (summed)
QMAP = {"windows": ["window"], "doors": ["door"], "toilets": ["toilet"], "sinks": ["sink"],
        "showers": ["bathtub", "shower"], "saunas": ["sauna"], "kitchens": ["room_kitchen"],
        "bedrooms": ["room_bedroom"], "bathrooms": ["room_bath"], "balconies": ["room_outdoor"]}
_lock = threading.Lock()  # YOLO model / HJ globals are not thread-safe


def _import_hj():
    if str(HJ_REPO) not in sys.path:
        sys.path.insert(0, str(HJ_REPO))
    from fp import infer, query  # noqa: E402
    return infer, query


def _to_original(poly, info):
    """Map a polygon from HJ preprocessed image coords back to the original image (undo scale, crop, deskew)."""
    import cv2
    p = np.asarray(poly, dtype=np.float64)
    if p.size == 0:
        return p
    p = p / info.get("scale", 1.0)
    if "crop" in info:
        p = p + np.array(info["crop"][:2], dtype=np.float64)
    ang = info.get("deskew_deg", 0.0) or 0.0
    if abs(ang) > 0.2:
        w, h = info["orig_size"]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        Mi = cv2.invertAffineTransform(M)
        p = np.hstack([p, np.ones((len(p), 1))]) @ Mi.T
    return p


def detect(path) -> dict:
    path = Path(path)
    cdir = config.CACHE_DIR / "hj"
    cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / f"{path.stem}.json"
    if cfile.exists():
        return json.loads(cfile.read_text())
    with _lock:
        infer, query = _import_hj()
        with tempfile.TemporaryDirectory() as td:
            sc = infer.run(path, Path(td), model=str(HJ_MODEL), auto_scale=False)
        info = {k: sc.meta.get(k) for k in ("orig_size", "deskew_deg", "crop", "scale") if sc.meta.get(k) is not None}
        by_id = {e.id: e for e in sc.objects + sc.rooms}
        out = {"counts": {}, "levels": {}, "boxes": {}, "info": info}
        for qt, classes in QMAP.items():
            n, lv, boxes = 0, "high", []
            order = {"low": 0, "medium": 1, "high": 2}
            for c in classes:
                a = query.count(sc, c)
                n += int(a.answer or 0)
                lv = min(lv, a.confidence or "low", key=order.get)
                for i in a.ids:
                    e = by_id.get(i)
                    if e is None:
                        continue
                    p = _to_original(e.polygon, info)
                    boxes.append({"box": [int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()),
                                          int(p[:, 1].max())], "score": round(float(e.conf), 3), "label": c})
            out["counts"][qt], out["levels"][qt], out["boxes"][qt] = n, lv, boxes
    cfile.write_text(json.dumps(out))
    return out


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser(description="cache HJ detections for images")
    ap.add_argument("--list")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    files = [config.DATASET_DIR / l.strip() for l in open(a.list) if l.strip()] if a.list else [Path(f) for f in a.files]
    t0 = time.time()
    for i, f in enumerate(files):
        r = detect(f)
        print(f"[{i + 1}/{len(files)}] {f.name} windows={r['counts']['windows']} doors={r['counts']['doors']} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
