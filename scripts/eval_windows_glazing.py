"""Окна: модель против «модель + полосы остекления» на вопросах об окнах eval_set Stoa.

    .venv/bin/python scripts/eval_windows_glazing.py --set "<eval_set>" --scenes runs/eval_stoa/run2/scenes [--split test]
Сцены не меняет.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def one(args):
    import cv2
    from fp.infer import add_glazing_windows
    from fp.scene import Scene
    d, expected = args
    sc = Scene.load(Path(d))
    n_model = sum(1 for o in sc.objects if o.cls == "window")
    info = add_glazing_windows(cv2.imread(str(Path(d) / "image.png")), sc)
    return Path(d).name, expected, n_model, n_model + info["added"], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--split", default="trainval")
    a = ap.parse_args()
    jobs = []
    for line in open(Path(a.set) / "questions.jsonl"):
        q = json.loads(line)
        if q["qtype"] != "windows" or not q["question"].startswith("How"):
            continue
        if (a.split == "test") != bool(q["cubicasa_test_split"]):
            continue
        jobs.append((str(Path(a.scenes) / Path(q["image"]).stem), q["answer"]))
    ok_m = ok_g = 0
    with ProcessPoolExecutor(4) as ex:
        for name, exp, nm, ng, info in ex.map(one, jobs):
            ok_m += nm == exp
            ok_g += ng == exp
            print(f"{name:40} ожид {exp:2}  модель {nm:2}  +полосы {ng:2}  (полос {info['bands']}, "
                  f"подтв {len(info['confirmed'])})", flush=True)
    n = len(jobs)
    print(f"\n{a.split}: {n} планов. Модель {ok_m}/{n} = {100 * ok_m / n:.1f}%,  +полосы {ok_g}/{n} = {100 * ok_g / n:.1f}%")


if __name__ == "__main__":
    main()
