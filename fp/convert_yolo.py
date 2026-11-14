"""CubiCasa5K -> датасет Ultralytics YOLO-seg.

    python -m fp.convert_yolo --src data/cubicasa5k --out data/yolo --image F1_scaled.png

Для каждой папки с model.svg пишет labels/<split>/<name>.txt со строками
`class_id x1 y1 x2 y2 ...` (нормировано 0..1) и симлинк на картинку в images/<split>/.
Сплиты берутся из train.txt/val.txt/test.txt датасета, если они есть.
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from pathlib import Path

import cv2

from .classes import CLASS_ID, CLASSES
from .cubicasa_svg import parse_cubicasa_svg


def find_samples(src: Path, image_name: str):
    for svg in sorted(src.rglob("model.svg")):
        img = svg.parent / image_name
        if not img.exists():
            alt = [p for p in svg.parent.glob("F1_*.png")]
            if not alt:
                continue
            img = alt[0]
        yield svg.parent, svg, img


def read_splits(src: Path) -> dict[str, str]:
    """folder(rel path) -> split, по файлам train.txt / val.txt / test.txt."""
    m = {}
    for split in ("train", "val", "test"):
        f = src / f"{split}.txt"
        if f.exists():
            for line in f.read_text().split():
                m[line.strip("/")] = split
    return m


def sample_name(folder: Path, src: Path) -> str:
    return str(folder.relative_to(src)).strip("/").replace(os.sep, "__")


def svg_to_image_scale(folder: Path, img: Path, w: int, h: int) -> tuple[float, float]:
    """Координаты model.svg = пиксели F1_scaled.png (width/height SVG — лишь габарит содержимого!).
    Для другой картинки (F1_original) — пересчёт через размер F1_scaled."""
    ref = folder / "F1_scaled.png"
    if img.name == ref.name or not ref.exists():
        return 1.0, 1.0
    rh, rw = cv2.imread(str(ref), cv2.IMREAD_UNCHANGED).shape[:2]
    return w / rw, h / rh


def convert_one(svg: Path, img: Path) -> tuple[list[str], Counter, tuple[int, int]]:
    im = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"не читается {img}")
    h, w = im.shape[:2]
    parsed = parse_cubicasa_svg(svg)
    sx, sy = svg_to_image_scale(svg.parent, img, w, h)
    lines = []
    for cls, poly in parsed.objects:
        pts = []
        for x, y in poly:
            pts += [min(max(x * sx / w, 0.0), 1.0), min(max(y * sy / h, 0.0), 1.0)]
        xs, ys = pts[0::2], pts[1::2]
        if max(xs) - min(xs) < 1e-4 or max(ys) - min(ys) < 1e-4:
            continue  # вырожденный полигон
        lines.append(f"{CLASS_ID[cls]} " + " ".join(f"{v:.6f}" for v in pts))
    return lines, parsed.unknown, (w, h)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--image", default="F1_scaled.png")
    ap.add_argument("--limit", type=int, default=0, help="для быстрых проверок")
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args(argv)

    src, out = a.src.resolve(), a.out.resolve()
    splits = read_splits(src)
    rnd = random.Random(0)
    stats, unknown, n = Counter(), Counter(), 0
    for folder, svg, img in find_samples(src, a.image):
        rel = str(folder.relative_to(src)).strip("/")
        split = splits.get(rel) or ("val" if rnd.random() < a.val_frac else "train")
        name = sample_name(folder, src)
        try:
            lines, unk, _ = convert_one(svg, img)
        except Exception as e:  # битые файлы в датасете бывают
            print(f"[skip] {rel}: {e}")
            continue
        unknown.update(unk)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
        link = out / "images" / split / f"{name}{img.suffix}"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(img.resolve())
        for ln in lines:
            stats[CLASSES[int(ln.split()[0])]] += 1
        n += 1
        if a.limit and n >= a.limit:
            break

    have = sorted({p.name for p in (out / "images").iterdir()}) if (out / "images").exists() else []
    yaml = [f"path: {out}"]
    yaml += [f"{s}: images/{s}" for s in ("train", "val", "test") if s in have]
    yaml += ["names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (out / "floorplans.yaml").write_text("\n".join(yaml) + "\n")

    print(f"планов: {n}")
    for c in CLASSES:
        print(f"  {c:14s} {stats[c]}")
    if unknown:
        print("не смаппилось (топ-30):")
        for k, v in unknown.most_common(30):
            print(f"  {k}: {v}")
    print(f"yaml: {out / 'floorplans.yaml'}")


if __name__ == "__main__":
    main()
