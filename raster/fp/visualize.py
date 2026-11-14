"""Визуальная проверка YOLO-разметки: рисует полигоны поверх картинок.

    python -m fp.visualize --data data/yolo --split train --n 20 --out out/check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES, COLORS


def hex2bgr(h: str):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def draw_labels(img: np.ndarray, label_text: str) -> np.ndarray:
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    fill = img.copy()
    for line in label_text.strip().splitlines():
        v = line.split()
        cls = CLASSES[int(v[0])]
        pts = np.array([[float(v[i]) * w, float(v[i + 1]) * h] for i in range(1, len(v) - 1, 2)], np.int32)
        color = hex2bgr(COLORS[cls])
        cv2.fillPoly(fill, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
    return cv2.addWeighted(fill, 0.35, img, 0.65, 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((a.data / "images" / a.split).iterdir())[: a.n]
    for p in imgs:
        lbl = a.data / "labels" / a.split / (p.stem + ".txt")
        img = cv2.imread(str(p))
        vis = draw_labels(img, lbl.read_text() if lbl.exists() else "")
        cv2.imwrite(str(a.out / f"{p.stem}.jpg"), vis)
    print(f"сохранено {len(imgs)} картинок в {a.out}")


if __name__ == "__main__":
    main()
