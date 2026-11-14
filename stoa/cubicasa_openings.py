"""CubiCasa original post-processing (junction heatmaps -> opening polygons), with a per-image time limit.

python -m stoa.cubicasa_openings --list reports/scale_subset.txt [--timeout 20]
Saves cache/cubicasa_polys/<stem>.json: {"windows": [boxes], "doors": [boxes], "icons": {cls: [boxes]}} or {"error"}.
"""
import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from . import config
from .cubicasa_infer import SPLIT, predict


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def boxes_of(pols):
    out = []
    for p in pols:
        a = np.asarray(p)
        out.append([int(a[:, 0].min()), int(a[:, 1].min()), int(a[:, 0].max()), int(a[:, 1].max())])
    return out


def process(f: Path, out_dir: Path, timeout: int):
    from floortrans.post_prosessing import get_polygons, split_prediction
    pred = predict(f, tta=True)
    H, W = pred.shape[2:]
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        heat, rms, ics = split_prediction(pred, (H, W), SPLIT)
        pols, types, _, _ = get_polygons((heat, rms, ics), 0.2, [1, 2])
        signal.alarm(0)
        rec = {"windows": [], "doors": [], "icons": {}}
        for p, t in zip(pols, types):
            b = boxes_of([p])[0]
            if t["type"] == "wall":
                continue
            if t["type"] == "icon":
                rec["icons"].setdefault(str(t["class"]), []).append(b)
            # openings: class 1 = window, 2 = door
            if t["type"] == "icon" and False:
                pass
        # openings are returned with type "icon"? keep raw for inspection
        rec["raw_types"] = [t for t in types if t["type"] != "wall"]
        rec["raw_boxes"] = [boxes_of([p])[0] for p, t in zip(pols, types) if t["type"] != "wall"]
    except Timeout:
        rec = {"error": "timeout"}
    except Exception as e:
        signal.alarm(0)
        rec = {"error": repr(e)}
    (out_dir / f"{f.stem}.json").write_text(json.dumps(rec, default=int))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list")
    ap.add_argument("--timeout", type=int, default=20)
    a = ap.parse_args()
    out = config.CACHE_DIR / "cubicasa_polys"
    out.mkdir(parents=True, exist_ok=True)
    files = ([config.DATASET_DIR / l.strip() for l in open(a.list) if l.strip()] if a.list
             else sorted((config.DATASET_DIR / "data" / "cubicasa5k").glob("*.png")))
    t0 = time.time()
    for i, f in enumerate(files):
        if (out / f"{f.stem}.json").exists():
            continue
        r = process(f, out, a.timeout)
        print(f"[{i+1}/{len(files)}] {f.name} {'ERR ' + r['error'] if 'error' in r else len(r['raw_types'])} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
