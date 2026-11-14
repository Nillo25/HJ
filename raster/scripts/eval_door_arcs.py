"""Двери: модель против «модель + дуги открывания» на вопросах о дверях eval_set Stoa.

    .venv/bin/python scripts/eval_door_arcs.py --set "<eval_set>" --scenes runs/eval_stoa/run2/scenes [--split trainval|test]
Сцены не меняет: дуги считаются на копии в памяти.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def one(args):
    import cv2
    from fp.infer import add_arc_doors
    from fp.scene import Scene
    d, expected = args
    sc = Scene.load(Path(d))
    n_model = sum(1 for o in sc.objects if o.cls == "door")
    t = time.time()
    info = add_arc_doors(cv2.imread(str(Path(d) / "image.png")), sc)
    n_arc = sum(1 for o in sc.objects if o.cls == "door")
    return Path(d).name, expected, n_model, n_arc, info, round(time.time() - t, 1), sc.meta.get("px_per_m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    for line in open(Path(a.set) / "questions.jsonl"):
        q = json.loads(line)
        if q["qtype"] != "doors" or not q["question"].startswith("How"):
            continue
        if (a.split == "test") != bool(q["cubicasa_test_split"]):
            continue
        jobs.append((str(Path(a.scenes) / Path(q["image"]).stem), q["answer"]))
    ok_m = ok_a = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for name, exp, nm, na, info, sec, ppm in ex.map(one, jobs):
            ok_m += nm == exp
            ok_a += na == exp
            print(f"{name:40} ожид {exp:2}  модель {nm:2}  +дуги {na:2}  ({info.get('arcs')} дуг, {info.get('near_walls')} у стен, "
                  f"+{info.get('added')}, {sec}s, ppm {ppm and round(ppm)})", flush=True)
    n = len(jobs)
    print(f"\n{a.split}: {n} планов. Модель {ok_m}/{n} = {100 * ok_m / n:.1f}%,  модель+дуги {ok_a}/{n} = {100 * ok_a / n:.1f}%")


if __name__ == "__main__":
    main()
