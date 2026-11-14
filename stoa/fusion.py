"""P2: multi-scale fusion of CubiCasa detections (windows, doors, other icons).

Views = cached inferences at several input scales (each already 4-rotation TTA averaged):
  cache/cubicasa (x1), cache/cubicasa_s1.5, cache/cubicasa_s2.

Two fusion modes:
  "vote"  - boxes from every view mapped to original pixels, clustered by IoS (intersection / smaller box);
            an object counts if supported by >= k views.  (Weighted-boxes-fusion style, box = score-weighted mean.)
  "pmean" - per-class probability maps (prob where argmax==class) resized to original size and averaged;
            threshold -> connected components.
Optional "merge": join collinear fragments of the same class whose gap along the wall axis is < gap_m metres.

python -m stoa.fusion            # grid search on non-test plans, report test
"""
import csv
import itertools
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import config, cubicasa_count as cc

VIEWS = ["cubicasa", "cubicasa_s1.5", "cubicasa_s2"]
ICON = {"windows": cc.WINDOW, "doors": cc.DOOR, "toilets": cc.TOILET, "sinks": cc.SINK, "showers": cc.BATHTUB}


def has_view(stem, v):
    return (config.CACHE_DIR / v / f"{stem}.npz").exists()


@lru_cache(maxsize=4096)
def view_boxes(stem, qtype, v):
    """objects of one view in ORIGINAL pixel coords: list of (x1,y1,x2,y2,score)."""
    d = cc.load(stem, v)
    sc = float(d["scale"])
    out = []
    for o in cc.objects(stem, qtype, v):
        # newer cubicasa_count returns "box" in original pixels (+ "mbox" in mask pixels); older one in mask pixels
        x1, y1, x2, y2 = o["box"] if "mbox" in o else [c / sc for c in o["box"]]
        out.append((x1, y1, x2, y2, o["score"]))
    return tuple(out)


def ios(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    sa = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    sb = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return inter / min(sa, sb)


def cluster(views_boxes, thr):
    """greedy WBF-like clustering. views_boxes: list (per view) of box lists. returns list of (box, support, score)."""
    allb = [(b, vi) for vi, bs in enumerate(views_boxes) for b in bs]
    allb.sort(key=lambda t: -t[0][4])
    clusters = []  # each: dict(box, members:[(b,vi)])
    for b, vi in allb:
        best, bi = 0.0, -1
        for ci, c in enumerate(clusters):
            if vi in c["views"]:
                continue
            s = ios(c["box"], b)
            if s > best:
                best, bi = s, ci
        if best >= thr:
            c = clusters[bi]
            c["m"].append(b)
            c["views"].add(vi)
            w = np.array([m[4] for m in c["m"]])
            c["box"] = tuple((np.array([m[:4] for m in c["m"]]) * w[:, None]).sum(0) / w.sum()) + (float(w.mean()),)
        else:
            clusters.append({"box": b, "m": [b], "views": {vi}})
    return [(c["box"], len(c["views"]), c["box"][4]) for c in clusters]


def merge_collinear(boxes, gap_px):
    """join boxes lying on the same axis (thin, aligned) with gap < gap_px. boxes: list of (x1,y1,x2,y2,...)."""
    if gap_px <= 0 or len(boxes) < 2:
        return list(boxes)
    bs = [list(b[:4]) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i, j in itertools.combinations(range(len(bs)), 2):
            a, b = bs[i], bs[j]
            ha = (a[2] - a[0]) >= (a[3] - a[1])
            hb = (b[2] - b[0]) >= (b[3] - b[1])
            if ha != hb:
                continue
            if ha:  # horizontal: y ranges overlap, x gap small
                ov = min(a[3], b[3]) - max(a[1], b[1])
                gap = max(a[0], b[0]) - min(a[2], b[2])
                thick = min(a[3] - a[1], b[3] - b[1])
            else:
                ov = min(a[2], b[2]) - max(a[0], b[0])
                gap = max(a[1], b[1]) - min(a[3], b[3])
                thick = min(a[2] - a[0], b[2] - b[0])
            if ov >= 0.5 * thick and gap < gap_px:
                bs[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                del bs[j]
                changed = True
                break
    return bs


@lru_cache(maxsize=512)
def class_prob_orig(stem, v, cls):
    d = cc.load(stem, v)
    p = np.where(d["icon_cls"] == cls, d["icon_prob"].astype(np.float32) / 255.0, 0.0).astype(np.float32)
    return p


def orig_size(stem):
    d = cc.load(stem, "cubicasa")
    return d["icon_cls"].shape  # scale 1 cache is original size


def pmean_count(stem, qtype, views, thr, min_frac):
    H, W = orig_size(stem)
    acc = np.zeros((H, W), np.float32)
    for v in views:
        p = class_prob_orig(stem, v, ICON[qtype])
        acc += cv2.resize(p, (W, H), interpolation=cv2.INTER_AREA)
    acc /= len(views)
    d = cc.load(stem, "cubicasa")
    rc = d["room_cls"]
    bld = max(int(((rc != cc.BG) & (rc != cc.OUTDOOR)).sum()), 1)
    m = (acc >= thr).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    amin = max(cc.P["icon_min_px"], min_frac * bld)
    return [(st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3]) for i in range(1, n) if st[i, 4] >= amin]


def ppm(stem):
    v = float(cc.load(stem, "cubicasa")["px_per_m"])
    return v if v > 0 else 45.0


def fused(stem, qtype, mode="vote", views=None, k=2, thr=0.3, gap_m=0.0, pthr=0.35, min_frac=0.0004,
          ppm_thr=30.0, hi_view="cubicasa_s2"):
    views = [v for v in (views or VIEWS) if has_view(stem, v)]
    if mode == "adapt":  # per-plan scale: small drawings (low px/m) -> upscaled view, else x1
        v = hi_view if ppm(stem) < ppm_thr and has_view(stem, hi_view) else "cubicasa"
        return merge_collinear([b[:4] for b in view_boxes(stem, qtype, v)], gap_m * ppm(stem))
    if mode == "single":
        boxes = [b[:4] for b in view_boxes(stem, qtype, views[0])]
    elif mode == "vote":
        k_eff = min(k, len(views))
        cl = cluster([view_boxes(stem, qtype, v) for v in views], thr)
        boxes = [c[0][:4] for c in cl if c[1] >= k_eff]
    elif mode == "pmean":
        boxes = pmean_count(stem, qtype, tuple(views), pthr, min_frac)
    else:
        raise ValueError(mode)
    return merge_collinear(boxes, gap_m * ppm(stem))


# ----------------------------------------------------------------------------------------------- evaluation
def load_gold():
    rows = list(csv.DictReader(open(config.REPORTS_DIR / "bench_cubicasa.csv")))
    return [(Path(r["path"]).stem, r["qtype"], int(r["expected"]), r["in_test"] == "True") for r in rows]


def grid():
    yield ("x1 (baseline)", dict(mode="single", views=["cubicasa"]))
    yield ("x1.5", dict(mode="single", views=["cubicasa_s1.5"]))
    yield ("x2", dict(mode="single", views=["cubicasa_s2"]))
    for k, thr in itertools.product([1, 2, 3], [0.2, 0.4]):
        yield (f"vote k={k} ios={thr}", dict(mode="vote", k=k, thr=thr))
    for pthr in [0.2, 0.3, 0.4]:
        yield (f"pmean p>={pthr}", dict(mode="pmean", pthr=pthr))
    for t, hv in itertools.product([20, 25, 30, 35, 40], ["cubicasa_s1.5", "cubicasa_s2"]):
        yield (f"adapt ppm<{t} {hv[-4:]}", dict(mode="adapt", ppm_thr=t, hi_view=hv))
    for g in [0.1, 0.3]:
        yield (f"vote k=2 +merge {g}m", dict(mode="vote", k=2, thr=0.4, gap_m=g))
        yield (f"x1 +merge {g}m", dict(mode="single", views=["cubicasa"], gap_m=g))


def evaluate(qtypes=("windows", "doors", "toilets", "sinks")):
    gold = [g for g in load_gold() if g[1] in qtypes and all(has_view(g[0], v) for v in VIEWS)]
    res = {}
    for name, kw in grid():
        for q in qtypes:
            for split in (False, True):
                gs = [g for g in gold if g[1] == q and g[3] == split]
                errs = [len(fused(s, q, **kw)) - e for s, _, e, _ in gs]
                res[(name, q, split)] = (np.mean([e == 0 for e in errs]), np.mean(errs), len(errs))
    return res


def main():
    qtypes = sys.argv[1:] or ["windows", "doors", "toilets", "sinks"]
    res = evaluate(qtypes)
    names = [n for n, _ in grid()]
    for q in qtypes:
        print(f"\n== {q}: acc / bias   [train = not in_test | TEST]")
        best = max(names, key=lambda n: res[(n, q, False)][0])
        for n in names:
            a, b, c = res[(n, q, False)]
            ta, tb, tc = res[(n, q, True)]
            mark = "  <- best on train" if n == best else ""
            print(f"  {n:24s} train {a:5.1%} {b:+5.2f} (n={c:3d}) | TEST {ta:5.1%} {tb:+5.2f} (n={tc}){mark}")


if __name__ == "__main__":
    main()
