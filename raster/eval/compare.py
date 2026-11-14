#!/usr/bin/env python3
"""Compare another solution's answers with ours on eval_set. Python 3.8+, standard library only.

Usage:
    python3 compare.py predictions.jsonl [--name colleague] [--csv out.csv]

predictions.jsonl - one JSON object per line:
    {"image": "colorful-11397.png", "question": "How many windows are shown on this floor plan?", "answer": 13}
  - "image": file name or any path ending in the file name (data/cubicasa5k/colorful-11397.png is fine)
  - "question": exact text from questions.jsonl (English or Finnish)
  - "answer": number (int, or string that parses as a number); null/missing = no answer (counted as wrong)
Questions are matched by (image file name, question text). For each (image, question type) the English
answer is used if present, otherwise the Finnish one. Scoring: exact match, as in the Stoa rules.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["windows", "doors", "toilets", "sinks", "showers", "saunas", "kitchens", "bedrooms", "wc_rooms",
         "bathrooms", "balconies", "floors"]
OURS = [("cubicasa", "CubiCasa"), ("haiku_5_5", "Haiku 5.5"), ("opus_5_5", "Opus 5.5"), ("hybrid", "Hybrid")]


def to_num(v):
    if v is None:
        return None
    try:
        f = float(str(v).strip())
    except ValueError:
        return None
    return int(f) if f.is_integer() else f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--name", default="theirs")
    ap.add_argument("--csv", help="write per-question comparison to this CSV")
    a = ap.parse_args()

    # questions: (image, question) -> (qtype, lang, answer, test)
    qs = {}
    for line in open(os.path.join(HERE, "questions.jsonl"), encoding="utf-8"):
        q = json.loads(line)
        img = os.path.basename(q["path"])
        lang = "en" if q["question"].startswith("How") else "fi"
        qs[(img, q["question"].strip())] = (q["qtype"], lang, q["answer"], q["cubicasa_test_split"])

    preds, unknown, dup = {}, 0, 0
    for i, line in enumerate(open(a.predictions, encoding="utf-8")):
        if not line.strip():
            continue
        p = json.loads(line)
        key = (os.path.basename(p.get("image") or p.get("path") or ""), (p.get("question") or "").strip())
        if key not in qs:
            unknown += 1
            continue
        dup += key in preds
        preds[key] = to_num(p.get("answer"))

    # one answer per (image, qtype): English preferred
    theirs = {}
    for (img, text), (qt, lang, ans, test) in qs.items():
        if (img, text) in preds:
            k = (img, qt)
            if k not in theirs or lang == "en":
                theirs[k] = preds[(img, text)]

    ours = {(r["image"], r["qtype"]): r for r in csv.DictReader(open(os.path.join(HERE, "our_results.csv"), encoding="utf-8"))}
    total = len(ours)
    answered = sum(1 for k in ours if theirs.get(k) is not None)
    print(f"Matched predictions: {len(preds)} lines; unknown (image/question not in set): {unknown}; duplicates: {dup}")
    print(f"Coverage: {answered}/{total} (image, question type) pairs have an answer; missing ones count as wrong\n")

    cols = [(c, n) for c, n in OURS] + [("__theirs", a.name)]

    def correct(r, col):
        if col == "__theirs":
            v = theirs.get((r["image"], r["qtype"]))
        else:
            v = to_num(r[col])
        return v is not None and v == to_num(r["expected"])

    def cell(rows, col):
        return f"{100 * sum(correct(r, col) for r in rows) / len(rows):5.1f}%" if rows else "    -"

    rows_all = list(ours.values())
    hdr = f"| {'type':10s} | {'n all/test':10s} | " + " | ".join(f"{n:^15s}" for _, n in cols) + " |"
    print("Accuracy, exact match: ALL 80 plans / CubiCasa TEST split (27 plans)\n")
    print(hdr)
    print("|" + "|".join("-" * (len(x)) for x in hdr.split("|")[1:-1]) + "|")
    for t in ORDER + ["TOTAL"]:
        a_ = [r for r in rows_all if t == "TOTAL" or r["qtype"] == t]
        te = [r for r in a_ if r["cubicasa_test_split"] == "True"]
        print(f"| {t:10s} | {len(a_):4d}/{len(te):<5d} | " + " | ".join(f"{cell(a_, c)} / {cell(te, c)}" for c, _ in cols) + " |")

    # head-to-head vs hybrid
    both = sum(correct(r, "hybrid") and correct(r, "__theirs") for r in rows_all)
    only_h = sum(correct(r, "hybrid") and not correct(r, "__theirs") for r in rows_all)
    only_t = sum(correct(r, "__theirs") and not correct(r, "hybrid") for r in rows_all)
    print(f"\nHead-to-head vs Hybrid (all {total}): both right {both}, only Hybrid right {only_h}, "
          f"only {a.name} right {only_t}, both wrong {total - both - only_h - only_t}")
    print("Note: test cells have 6-27 questions each; one answer moves them by 4-15 points.")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(rows_all[0].keys()) + [a.name])
            for r in rows_all:
                w.writerow(list(r.values()) + [theirs.get((r["image"], r["qtype"]))])
        print(f"Per-question CSV: {a.csv}")


if __name__ == "__main__":
    sys.exit(main())
