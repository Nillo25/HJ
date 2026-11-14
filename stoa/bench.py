"""Evaluation bench: accuracy by question type x plan subtype, cost, share answered without VLM.

python -m stoa.bench --method gemini_batch [--images 40] [--lang en|all] [--workers 8]
"""
import argparse
import collections
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from . import config
from .answer import answer
from .qtypes import QTYPES, qtype_of, subtype_of

SUBS = ["hq_arch", "hq", "colorful"]


def load_questions(lang="en"):
    rows = []
    for line in open(config.DATASET_DIR / "questions.jsonl"):
        q = json.loads(line)
        if "/cubicasa5k/" not in q["path"]:
            continue
        if lang == "en" and not q["question"].startswith("How"):
            continue
        q["qtype"] = qtype_of(q["question"])
        q["subtype"] = subtype_of(q["path"])
        rows.append(q)
    return rows


def test_split():
    """Image names (e.g. high_quality_architectural-1000) that are in CubiCasa5K test.txt."""
    if not config.CUBICASA_TEST_LIST.exists():
        return None
    s = set()
    for line in config.CUBICASA_TEST_LIST.read_text().split():
        parts = line.strip("/").split("/")  # /high_quality_architectural/1000/
        if len(parts) >= 2:
            s.add(f"{parts[-2]}-{parts[-1]}")
    return s


def fmt_table(res, key_rows, key_cols, title):
    cols = key_cols + ["ALL"]
    out = [title, f"{'':12s}" + "".join(f"{c:>14s}" for c in cols)]
    for r in key_rows + ["ALL"]:
        cells = []
        for c in cols:
            sel = [x for x in res if (r == "ALL" or x["qtype"] == r) and (c == "ALL" or x["subtype"] == c)]
            if not sel:
                cells.append(f"{'-':>14s}")
                continue
            acc = sum(x["correct"] for x in sel) / len(sel)
            cells.append(f"{acc*100:6.1f}% ({len(sel):4d})")
        out.append(f"{r:12s}" + "".join(cells))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="gemini_batch")
    ap.add_argument("--model", default=None)
    ap.add_argument("--images", type=int, default=0, help="random sample of N images (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prefer-test", action="store_true", help="sample includes all CubiCasa test images first")
    a = ap.parse_args()

    qs = load_questions(a.lang)
    imgs = sorted({q["path"] for q in qs})
    if a.images:
        random.Random(a.seed).shuffle(imgs)
        if a.prefer_test and test_split():
            t = test_split()
            imgs.sort(key=lambda p: p.rsplit("/", 1)[-1][:-4] not in t)  # stable: test first
        imgs = set(imgs[: a.images])
        qs = [q for q in qs if q["path"] in imgs]
    test = test_split()

    kw = {"model": a.model} if a.model and a.method in ("gemini_batch", "gemini_single", "claude", "hybrid") else {}
    by_img = collections.defaultdict(list)
    for q in qs:
        by_img[q["path"]].append(q)

    def run_img(p):
        out = []
        for q in by_img[p]:
            try:
                r = answer(p, q["question"], a.method, **kw)
            except Exception as e:
                print(f"ERR {p} {q['qtype']}: {e}", file=sys.stderr)
                r = {"answer": None, "cost_usd": 0, "answered_by": "error", "evidence": [], "confidence": "low",
                     "call_cost_shared": False, "parsed_by": "error"}
            out.append((q, r))
        return out

    t0 = time.time()
    res = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, out in enumerate(ex.map(run_img, list(by_img))):
            for q, r in out:
                name = q["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
                res.append({
                    "path": q["path"], "qtype": q["qtype"], "subtype": q["subtype"],
                    "in_test": (name in test) if test is not None else "",
                    "expected": q["answer"], "got": r["answer"], "correct": r["answer"] == q["answer"],
                    "confidence": r["confidence"], "answered_by": r["answered_by"],
                    "n_evidence": len(r["evidence"]), "cost_usd": r["cost_usd"],
                    "shared": r.get("call_cost_shared", False), "parsed_by": r["parsed_by"],
                    "route": r.get("route", a.method),
                    "plan_reliable": (r.get("plan_check") or {}).get("reliable", ""),
                    "question": q["question"],
                })
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(by_img)} images, {time.time()-t0:.0f}s", file=sys.stderr)

    # cost: shared calls counted once per image
    seen, total = set(), 0.0
    for x in res:
        k = (x["path"], x["route"]) if x["shared"] else (x["path"], x["qtype"])
        if k not in seen:
            seen.add(k)
            total += x["cost_usd"]
    fresh = sum(1 for x in res if x["parsed_by"] == "llm")

    tag = a.method + (f"_{a.model}" if a.model else "") + (f"_{a.tag}" if a.tag else "")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    csv_path = config.REPORTS_DIR / f"bench_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)

    print(fmt_table(res, QTYPES, SUBS, f"\n== {tag}: accuracy, all plans (n={len(res)}, images={len(by_img)})"))
    if test is not None:
        tr = [x for x in res if x["in_test"]]
        if tr:
            print(fmt_table(tr, QTYPES, SUBS, f"\n== {tag}: accuracy, CubiCasa TEST split only (n={len(tr)})"))
    no_gem = sum(1 for x in res if x["answered_by"] in ("cubicasa", "detector")) / len(res)
    print(f"\nVLM cost (full, as if uncached): ${total:.4f} total, ${total/len(res):.5f}/answer, "
          f"${total/len(by_img):.5f}/image | fresh calls this run: {fresh} | answered without VLM: {no_gem*100:.0f}%")
    pr = [x for x in res if x["plan_reliable"] != ""]
    if pr:
        for scope, rows in (("all", pr), ("test", [x for x in pr if x["in_test"] is True])):
            rel = [x for x in rows if x["plan_reliable"]]
            unr = [x for x in rows if not x["plan_reliable"]]
            if rows:
                print(f"Plan check ({scope}): reliable plans {len({x['path'] for x in rel})}, accuracy "
                      f"{100*sum(x['correct'] for x in rel)/max(len(rel),1):.1f}% | UNRELIABLE plans "
                      f"{len({x['path'] for x in unr})}, accuracy {100*sum(x['correct'] for x in unr)/max(len(unr),1):.1f}%")
    for lv in ("high", "medium", "low"):
        sel = [x for x in res if x["confidence"] == lv]
        if sel:
            print(f"confidence {lv:6s}: {len(sel):4d} answers ({100*len(sel)/len(res):.0f}%), "
                  f"accuracy {100*sum(x['correct'] for x in sel)/len(sel):.1f}%")
    errs = [x for x in res if not x["correct"]]
    random.Random(1).shuffle(errs)
    print("\nSample errors:")
    for x in errs[:6]:
        print(f"  {x['path'].rsplit('/',1)[-1]:40s} {x['qtype']:10s} expected={x['expected']} got={x['got']}")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
