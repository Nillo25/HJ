"""Подбор порога уверенности по классам и склейки дублей — под метрику Stoa (точное совпадение числа).

Шаг 1 (модель, один раз): сохранить все детекции с низким порогом
    python scripts/tune_thresholds.py dump --yolo runs/colab/v1_best.pt --split val
    python scripts/tune_thresholds.py dump --yolo runs/colab/v1_best.pt --split test
Шаг 2 (без модели, секунды): подобрать на val, проверить на test
    python scripts/tune_thresholds.py tune

Подбираем на val, а цифры отчёта — на test, чтобы не подогнать пороги под тот же набор.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fp.cubicasa_svg import parse_cubicasa_svg  # noqa: E402

OUT = ROOT / "out/tune"
CONFS = [round(x, 2) for x in np.arange(0.05, 0.91, 0.05)]
MERGE = [None, 0.1, 0.3, 0.5]
MIN_GAIN = 0.02  # минимальный прирост точности на val, чтобы сменить стандартные настройки   # склеивать боксы одного класса, если пересечение / меньший бокс >= x


def dump(weights: str, split: str, device: str, data: Path):
    from ultralytics import YOLO
    model = YOLO(weights)
    imgs = sorted((data / "images" / split).glob("*.png"))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{split}.jsonl").open("w") as fh:
        for i, p in enumerate(imgs):
            r = model.predict(str(p), imgsz=1024, conf=0.05, max_det=1000, device=device, verbose=False)[0]
            dets = [[r.names[int(c)], round(float(s), 3), [round(v, 1) for v in b]]
                    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            gt = dict(Counter(c for c, _ in parse_cubicasa_svg(p.resolve().parent / "model.svg").objects))
            fh.write(json.dumps({"plan": p.stem, "gt": gt, "dets": dets}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{split}: {i + 1}/{len(imgs)}", flush=True)


def _ios(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / max(small, 1e-6)


def count(boxes: list, merge: float | None) -> int:
    """Число объектов после склейки: связные группы боксов с пересечением/меньший >= merge."""
    if merge is None or len(boxes) < 2:
        return len(boxes)
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _ios(boxes[i], boxes[j]) >= merge:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(boxes))})


def accuracy(rows, cls, conf, merge, subset=None) -> float:
    hits = []
    for r in rows:
        if subset and not r["plan"].startswith(subset + "__"):
            continue
        boxes = [b for c, s, b in r["dets"] if c == cls and s >= conf]
        hits.append(count(boxes, merge) == r["gt"].get(cls, 0))
    return float(np.mean(hits)) if hits else float("nan")


def tune():
    val = [json.loads(l) for l in (OUT / "val.jsonl").open()]
    test = [json.loads(l) for l in (OUT / "test.jsonl").open()]
    classes = sorted({c for r in val for c, _, _ in r["dets"]} | {c for r in val for c in r["gt"]})
    classes = [c for c in classes if c != "wall" and any(c == d[0] for r in val for d in r["dets"])]
    best = {}
    for cls in classes:
        grid = [(accuracy(val, cls, t, m), t, m) for t in CONFS for m in MERGE]
        acc, t, m = max(grid, key=lambda g: (g[0], -abs(g[1] - 0.25), g[2] is None))
        base = accuracy(val, cls, 0.25, None)
        # меньше MIN_GAIN на val — считаем шумом и оставляем стандарт (0.25, без склейки)
        best[cls] = {"conf": t, "merge": m} if acc - base >= MIN_GAIN else {"conf": 0.25, "merge": None}
        best[cls]["val_gain"] = round(acc - base, 3)
    lines = ["| класс | порог | склейка | test: было (0.25, без склейки) | test: стало | из них architectural: было → стало |",
             "|---|---|---|---|---|---|"]
    for cls in classes:
        b = best[cls]
        was, now = accuracy(test, cls, 0.25, None), accuracy(test, cls, b["conf"], b["merge"])
        aw = accuracy(test, cls, 0.25, None, "high_quality_architectural")
        an = accuracy(test, cls, b["conf"], b["merge"], "high_quality_architectural")
        lines.append(f"| {cls} | {b['conf']} | {b['merge'] or '—'} | {was:.0%} | {now:.0%} | {aw:.0%} → {an:.0%} |")
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text + "\n")
    (OUT / "thresholds.json").write_text(json.dumps(best, indent=1))
    print(text)
    print(f"\nпороги: {OUT / 'thresholds.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "tune", "calibrate"])
    ap.add_argument("--yolo")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--data", default=ROOT / "data/yolo", type=Path)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.yolo, a.split, a.device, a.data)
    elif a.cmd == "tune":
        tune()
    else:
        calibrate()



# ---------------- калибровка уверенности ----------------
NEAR_MARGIN = 0.15          # детекция «пограничная», если |conf - порог класса| < NEAR_MARGIN
HIGH_TARGET, MEDIUM_TARGET = 0.90, 0.70   # целевая доля верных ответов у high / medium


def signals(r, cls, t):
    confs = [s for c, s, _ in r["dets"] if c == cls and s >= t["conf"]]
    boxes = [b for c, s, b in r["dets"] if c == cls and s >= t["conf"]]
    near = sum(1 for c, s, _ in r["dets"] if c == cls and abs(s - t["conf"]) < NEAR_MARGIN)
    ok = count(boxes, t["merge"]) == r["gt"].get(cls, 0)
    return ok, near, (min(confs) if confs else 1.0)


def level_of(near, minconf, cal):
    if cal["high_minconf"] is not None and near == 0 and minconf >= cal["high_minconf"]:
        return "high"
    return "medium" if near <= cal["medium_max_near"] else "low"


def calibrate():
    th = json.loads((ROOT / "fp/thresholds.json").read_text())
    val = [json.loads(l) for l in (OUT / "val.jsonl").open()]
    test = [json.loads(l) for l in (OUT / "test.jsonl").open()]
    classes = [k for k in th if not k.startswith("_") and k != "default"]
    for cls in classes:
        t = th[cls]
        S = [signals(r, cls, t) for r in val]
        # high: near == 0 и minconf >= x; берём самый мягкий x, при котором точность high >= HIGH_TARGET
        high = None
        for x in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            grp = [ok for ok, n, m in S if n == 0 and m >= x]
            if len(grp) >= 20 and np.mean(grp) >= HIGH_TARGET:
                high = x
                break
        # medium: допускаем 1 пограничную, если такие ответы верны в >= MEDIUM_TARGET, иначе только 0
        one = [ok for ok, n, m in S if n == 1]
        t["high_minconf"] = high
        t["medium_max_near"] = 1 if one and np.mean(one) >= MEDIUM_TARGET else 0
    th["_calibration"] = {"near_margin": NEAR_MARGIN, "high_target": HIGH_TARGET, "medium_target": MEDIUM_TARGET}
    (ROOT / "fp/thresholds.json").write_text(json.dumps(th, indent=1, ensure_ascii=False))

    lines = ["| класс | high: доля / верно | medium: доля / верно | low: доля / верно |", "|---|---|---|---|"]
    tot = {"high": [], "medium": [], "low": []}
    for cls in classes:
        by = {"high": [], "medium": [], "low": []}
        for r in test:
            ok, n, m = signals(r, cls, th[cls])
            by[level_of(n, m, th[cls])].append(ok)
        for k in by:
            tot[k] += by[k]
        cell = lambda v: f"{len(v) / len(test):.0%} / {np.mean(v):.0%}" if v else "—"
        lines.append(f"| {cls} | {cell(by['high'])} | {cell(by['medium'])} | {cell(by['low'])} |")
    n = sum(len(v) for v in tot.values())
    lines.append(f"| **все классы** | {len(tot['high']) / n:.0%} / {np.mean(tot['high']):.0%} | "
                 f"{len(tot['medium']) / n:.0%} / {np.mean(tot['medium']):.0%} | {len(tot['low']) / n:.0%} / {np.mean(tot['low']):.0%} |")
    text = "\n".join(lines)
    (OUT / "calibration.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
