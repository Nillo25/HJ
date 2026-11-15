#!/usr/bin/env python3
"""Run the HJ floor-plan MVP (YOLO-seg + chat) on eval_set, exactly as its web UI does:
upload -> fp.infer.run(image, model) -> for each question: fp.chat.chat(scene, question).

Run with HJ's own interpreter:
    <HJ>/.venv/bin/python run_hj.py --repo <HJ> --model <HJ>/runs/colab/v2_best.pt [--lang en] [--limit N]
Writes predictions_hj.jsonl (for compare.py) and hj_raw.jsonl (full chat replies for debugging).
Resumable: already answered questions are skipped.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def number_from(res: dict):
    """Numeric answer: last count() tool result, else first number in the reply text."""
    for c in reversed(res.get("calls") or []):
        if c.get("tool") in ("count", "highlight"):
            v = (c.get("result") or {}).get("answer")
            if isinstance(v, (int, float)):
                return v
    m = re.search(r"-?\d+(?:[.,]\d+)?", res.get("reply") or "")
    return float(m.group().replace(",", ".")) if m else None


# --mode direct: HJ detector + HJ count(), but question type -> class mapped correctly (bypasses chat parsing)
DIRECT = {"windows": ["window"], "doors": ["door"], "toilets": ["toilet"], "sinks": ["sink"],
          "showers": ["bathtub", "shower"], "saunas": ["sauna"], "kitchens": ["room_kitchen"],
          "bedrooms": ["room_bedroom"], "bathrooms": ["room_bath"], "balconies": ["room_outdoor"],
          "wc_rooms": None, "floors": None}  # no matching class in HJ -> no answer


def direct(sc, qtype):
    from fp.query import count
    classes = DIRECT[qtype]
    if not classes:
        return {"reply": "no class", "calls": []}
    calls = [{"tool": "count", "args": {"cls": c}, "result": count(sc, c).to_dict()} for c in classes]
    total = sum(c["result"]["answer"] or 0 for c in calls)
    return {"reply": str(total), "calls": [{"tool": "count", "args": {"cls": "+".join(classes)},
                                             "result": {"answer": total}}], "parsed_by": "direct"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="en", choices=["en", "fi", "all"])
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    ap.add_argument("--mode", default="ui", choices=["ui", "direct"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="suffix for output files and scene cache (e.g. HJ commit)")
    a = ap.parse_args()
    t = f"_{a.tag}" if a.tag else ""
    a.out = a.out or str(HERE / (f"predictions_hj{t}.jsonl" if a.mode == "ui" else f"predictions_hj_direct{t}.jsonl"))
    raw_name = f"hj_raw{t}.jsonl" if a.mode == "ui" else f"hj_raw_direct{t}.jsonl"

    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    from fp import infer  # noqa: E402
    from fp.chat import chat  # noqa: E402

    qs = [json.loads(l) for l in open(HERE / "questions.jsonl", encoding="utf-8")]
    if a.lang != "all":
        qs = [q for q in qs if q["question"].startswith("How") == (a.lang == "en")]
    by_img = {}
    for q in qs:
        by_img.setdefault(q["image"], []).append(q)
    imgs = sorted(by_img)[: a.limit or None]

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            p = json.loads(l)
            done.add((p["image"], p["question"]))
    scenes = HERE / f"hj_scenes{t}"
    fo = open(a.out, "a", encoding="utf-8")
    fr = open(HERE / raw_name, "a", encoding="utf-8")
    cost = 0.0
    t0 = time.time()
    for i, img in enumerate(imgs):
        todo = [q for q in by_img[img] if (os.path.basename(img), q["question"]) not in done]
        if not todo:
            continue
        out_dir = scenes / Path(img).stem
        if (out_dir / "scene.json").exists():
            from fp.scene import Scene
            sc = Scene.load(out_dir)
        else:
            try:
                sc = infer.run(HERE / img, out_dir, model=a.model)
            except Exception as e:  # noqa: BLE001 — HJ bug in sheet/scale analysis; counting does not need it
                print(f"  upload failed ({type(e).__name__}: {e}); retry without auto scale", flush=True)
                with open(HERE / f"hj_upload_failures{t}.txt", "a") as ff:
                    ff.write(f"{img}\t{type(e).__name__}: {e}\n")
                sc = infer.run(HERE / img, out_dir, model=a.model, auto_scale=False)
        cost += float(sc.meta.get("sheet_cost_usd") or 0)
        for q in todo:
            try:
                res = chat(sc, q["question"]) if a.mode == "ui" else direct(sc, q["qtype"])
            except Exception as e:  # noqa: BLE001
                res = {"reply": f"ERROR {type(e).__name__}: {e}", "calls": []}
            ans = number_from(res)
            cost += float(res.get("cost_usd") or 0)
            fo.write(json.dumps({"image": os.path.basename(img), "question": q["question"], "answer": ans},
                                ensure_ascii=False) + "\n")
            fr.write(json.dumps({"image": os.path.basename(img), "qtype": q["qtype"], "question": q["question"],
                                 "expected": q["answer"], "answer": ans, "reply": res.get("reply"),
                                 "parsed_by": res.get("parsed_by"), "confidence": res.get("confidence"),
                                 "calls": [{"tool": c.get("tool"), "args": c.get("args"),
                                            "answer": (c.get("result") or {}).get("answer")}
                                           for c in res.get("calls") or []]},
                                ensure_ascii=False) + "\n")
            fo.flush()
            fr.flush()
        print(f"[{i + 1}/{len(imgs)}] {img}  ({time.time() - t0:.0f}s, LLM ${cost:.3f})", flush=True)
    print(f"done -> {a.out}; LLM cost this run ${cost:.3f}")


if __name__ == "__main__":
    main()
