"""Сравнение YOLO-seg (наша) и готовой модели CubiCasa5K по точности ПОДСЧЁТА на test-сплите.

Метрика как у Stoa: доля планов, где число объектов класса совпало с разметкой ТОЧНО (+ MAE).
Разметка — model.svg, разобранный нашим парсером (те же правила, что при обучении YOLO).

    python scripts/compare_models.py --yolo runs/colab/v1_best.pt --cubicasa data/ext/weights/model_best_val_loss_var.pkl \
        --repo data/ext/CubiCasa5k --out out/compare --n 400

Результаты по планам пишутся построчно в <out>/<model>.jsonl (можно прервать и продолжить).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fp.cubicasa_svg import parse_cubicasa_svg  # noqa: E402

# общие классы: наш класс -> (тип в CubiCasa, номер класса в CubiCasa)
CC_ICON = {"window": 1, "door": 2, "toilet": 5, "sink": 6, "sauna_bench": 7, "bathtub": 9}
CC_ROOM = {"room_outdoor": 1, "room_kitchen": 3, "room_living": 4, "room_bedroom": 5, "room_bath": 6,
           "room_entry": 7, "room_storage": 9}
COMPARE = list(CC_ICON) + ["shower"] + list(CC_ROOM)


def test_plans(data_yolo: Path, n: int):
    imgs = sorted((data_yolo / "images" / "test").glob("*.png"))[:n]
    return [(p.stem, p.resolve()) for p in imgs]  # симлинк -> .../F1_scaled.png


def gt_counts(img: Path) -> dict:
    parsed = parse_cubicasa_svg(img.parent / "model.svg")
    return dict(Counter(c for c, _ in parsed.objects))


# ---------------- YOLO ----------------
def run_yolo(weights: str, conf: float, device: str):
    from ultralytics import YOLO
    model = YOLO(weights)

    def f(img: Path) -> dict:
        r = model.predict(str(img), imgsz=1024, conf=conf, max_det=1000, device=device, verbose=False)[0]
        return dict(Counter(r.names[int(c)] for c in r.boxes.cls.tolist()))
    return f


# ---------------- CubiCasa ----------------
def run_cubicasa(weights: str, repo: str, device: str):
    import cv2
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, repo)
    from floortrans.models.hg_furukawa_original import hg_furukawa_original
    from floortrans.loaders.augmentations import RotateNTurns
    from floortrans.post_prosessing import split_prediction, get_polygons

    model = hg_furukawa_original(n_classes=51)  # без init_weights: всё равно перезаписываем весами
    model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
    ck = torch.load(weights, map_location="cpu", weights_only=True)  # только тензоры, без исполнения pickle
    model.load_state_dict(ck["model_state"])
    model.eval().to(device)
    rot = RotateNTurns()
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]  # как в samples.ipynb из репозитория
    inv_room = {v: k for k, v in CC_ROOM.items()}
    inv_icon = {v: k for k, v in CC_ICON.items()}

    def f(img: Path) -> dict:
        im = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        x = torch.tensor(2 * (np.moveaxis(im, -1, 0) / 255.0) - 1, dtype=torch.float32)[None].to(device)
        preds = torch.zeros([len(rotations), 44, h, w])
        with torch.no_grad():
            for i, (fw, back) in enumerate(rotations):
                p = model(rot(x, "tensor", fw))
                p = rot(p, "tensor", back)
                p = rot(p, "points", back)
                preds[i] = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=True)[0].cpu()
        pred = torch.mean(preds, 0, True)
        heat, rooms, icons = split_prediction(pred, (h, w), [21, 12, 11])
        polys, types, room_polys, room_types = get_polygons((heat, rooms, icons), 0.2, [1, 2])
        c = Counter()
        for t in types:
            if t["type"] == "icon" and int(t["class"]) in inv_icon:
                c[inv_icon[int(t["class"])]] += 1
        for t in room_types:
            if int(t["class"]) in inv_room:
                c[inv_room[int(t["class"])]] += 1
        return dict(c)
    return f


def evaluate(name: str, fn, plans, out: Path):
    path = out / f"{name}.jsonl"
    done = {json.loads(l)["plan"] for l in path.open()} if path.exists() else set()
    with path.open("a") as fh:
        for i, (plan, img) in enumerate(plans):
            if plan in done:
                continue
            t = time.time()
            try:
                pred = fn(img)
            except Exception as e:  # noqa: BLE001
                print(f"[{name}] {plan}: ошибка {e}", flush=True)
                pred = None
            fh.write(json.dumps({"plan": plan, "pred": pred, "sec": round(time.time() - t, 2)}) + "\n")
            fh.flush()
            print(f"[{name}] {i + 1}/{len(plans)} {plan} {time.time() - t:.1f}s", flush=True)


def report(out: Path, plans):
    gt = {p: gt_counts(img) for p, img in plans}
    rows, models = [], sorted(p.stem for p in out.glob("*.jsonl"))
    preds = {m: {d["plan"]: d["pred"] for d in map(json.loads, (out / f"{m}.jsonl").open())} for m in models}
    lines = ["| класс | планов с объектом | " + " | ".join(f"{m}: точно / ±1 / MAE" for m in models) + " |",
             "|---|---|" + "---|" * len(models)]
    for cls in COMPARE:
        cells = []
        common = [p for p, _ in plans if all(preds[m].get(p) is not None for m in models)]
        for m in models:
            if m.startswith("cubicasa") and cls == "shower":
                cells.append("— (нет класса)")
                continue
            if m.startswith("yolo") and cls == "sauna_bench" and not any(
                    cls in (preds[m][p] or {}) for p in common):
                cells.append("— (нет в v1)")
                continue
            err = np.array([preds[m][p].get(cls, 0) - gt[p].get(cls, 0) for p in common])
            cells.append(f"{np.mean(err == 0):.0%} / {np.mean(np.abs(err) <= 1):.0%} / {np.mean(np.abs(err)):.2f}")
        has = sum(gt[p].get(cls, 0) > 0 for p in common)
        lines.append(f"| {cls} | {has}/{len(common)} | " + " | ".join(cells) + " |")
    secs = {m: np.mean([d["sec"] for d in map(json.loads, (out / f"{m}.jsonl").open())]) for m in models}
    lines.append("")
    lines.append("Время на план: " + ", ".join(f"{m} {s:.1f} с" for m, s in secs.items()))
    text = "\n".join(lines)
    (out / "report.md").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/yolo"), type=Path)
    ap.add_argument("--yolo")
    ap.add_argument("--yolo-name", default="yolo_v1")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cubicasa")
    ap.add_argument("--repo", default=str(ROOT / "data/ext/CubiCasa5k"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=ROOT / "out/compare", type=Path)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plans = test_plans(a.data, a.n)
    if not a.report_only:
        if a.yolo:
            evaluate(a.yolo_name, run_yolo(a.yolo, a.conf, a.device), plans, a.out)
        if a.cubicasa:
            evaluate("cubicasa", run_cubicasa(a.cubicasa, a.repo, a.device), plans, a.out)
    report(a.out, plans)


if __name__ == "__main__":
    main()
