"""Добавить опись плана (fp.inventory) в уже разобранные сцены: meta["inventory"]. Без повторного YOLO и разбора листа.

    .venv/bin/python scripts/add_inventory.py runs/eval_stoa/run1/scenes [--workers 6] [--force]
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server.app  # noqa: E402,F401 — .env
import cv2  # noqa: E402
from fp.inventory import read_inventory  # noqa: E402
from fp.scene import Scene  # noqa: E402


def one(d: Path, force: bool):
    sc = Scene.load(d)
    if sc.meta.get("inventory") and not sc.meta["inventory"].get("error") and not force:
        return d.name, 0.0, "есть"
    inv, cost = read_inventory(cv2.imread(str(d / "image.png")))
    inv["cost_usd"] = round(cost, 6)
    sc.meta["inventory"] = inv
    sc.meta["sheet_cost_usd"] = round(sc.meta.get("sheet_cost_usd", 0) + cost, 6)
    sc.save(d)
    return d.name, cost, inv.get("error") or "ok"


ap = argparse.ArgumentParser()
ap.add_argument("scenes")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
dirs = sorted(p for p in Path(a.scenes).iterdir() if (p / "scene.json").exists())
total = 0.0
with ThreadPoolExecutor(a.workers) as ex:
    for n, (name, cost, st) in enumerate(ex.map(lambda d: one(d, a.force), dirs), 1):
        total += cost
        print(f"[{n}/{len(dirs)}] {name}: {st} ${cost:.4f}", flush=True)
print(f"итого ${total:.4f}")
