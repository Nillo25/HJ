"""Count window panes (sections between mullions) inside a detected window box.

Ground truth counts each sash/pane separated by a mullion as a separate window.
For a horizontal window, a mullion is a column of dark pixels spanning most of the frame thickness.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

PP = {"dark": 150, "frac": 0.7, "end_margin": 0.12, "max_run": 0.12, "shrink": 0.15}


@lru_cache(maxsize=16)
def gray(stem: str):
    return np.asarray(Image.open(config.DATASET_DIR / "data" / "cubicasa5k" / f"{stem}.png").convert("L"))


def count_panes(stem: str, box, scale: float = 1.0) -> int:
    g = gray(stem)
    x1, y1, x2, y2 = [int(round(v / scale)) for v in box]
    crop = g[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return 1
    if crop.shape[0] > crop.shape[1]:  # vertical window -> rotate so length is along axis 1
        crop = crop.T
    h, L = crop.shape
    if L < 8 or h < 3:
        return 1
    # use the inner band across the thickness (drop a bit of the outer box which may include wall/hatch)
    s = int(h * PP["shrink"])
    band = crop[s:h - s] if h - 2 * s >= 3 else crop
    frac = (band < PP["dark"]).mean(axis=0)
    on = frac >= PP["frac"]
    runs, i = [], 0
    while i < L:
        if on[i]:
            j = i
            while j < L and on[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    m = PP["end_margin"] * L
    mull = [r for r in runs if (r[0] + r[1]) / 2 > m and (r[0] + r[1]) / 2 < L - m and (r[1] - r[0]) <= PP["max_run"] * L]
    return 1 + len(mull)
