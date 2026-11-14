#!/usr/bin/env python3
"""Eval harness: run questions.jsonl and print the accuracy table.

    python eval.py <dataset folder> [--db app.db] [--source swiss_dwellings]
                   [--api http://localhost:3000/api/ask] [--limit 500]

Without --api it calls answer.ask directly: fast, works before the backend.
With --api it goes over HTTP, measuring what the judges will actually see.

Their scoring rules: counts exact, areas/lengths/volumes within 2%,
text free-form (skipped here, judged separately).
Language dedupe: the same question in 2-4 languages is one check,
otherwise the percentages are inflated 3.4x.
"""
import argparse
import collections
import json
import os
import sqlite3
import time
import uuid

import answer as A

TOL = {"area": 0.02, "length": 0.02, "volume": 0.02}


def verdict(expected, got, qtype) -> str:
    if got is None:
        return "refused"
    if qtype == "count":
        try:
            return "correct" if int(got) == int(expected) else "wrong"
        except (TypeError, ValueError):
            return "wrong"
    if qtype in TOL:
        try:
            e, g = float(expected), float(got)
        except (TypeError, ValueError):
            return "wrong"
        if e == 0:
            return "correct" if g == 0 else "wrong"
        return "correct" if abs(g - e) <= TOL[qtype] * abs(e) else "wrong"
    return "skipped"          # text / other: by hand or an LLM judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--db", default="app.db")
    ap.add_argument("--source", default=None)
    ap.add_argument("--api", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--show", type=int, default=12, help="how many misses to print")
    a = ap.parse_args()

    qpath = os.path.join(a.root, "questions.jsonl")
    rows = [json.loads(l) for l in open(qpath, encoding="utf-8") if l.strip()]
    if a.source:
        rows = [r for r in rows if f"/{a.source}/" in r["path"]]

    if not a.no_dedupe:
        seen, uniq = set(), []
        for r in rows:
            k = (r["path"], str(r["answer"]), r["type"], str(r.get("unit")))
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        print(f"questions {len(rows)} -> after language dedupe {len(uniq)}")
        rows = uniq
    if a.limit:
        rows = rows[: a.limit]

    con = sqlite3.connect(a.db)
    run_id = uuid.uuid4().hex[:8]
    stats = collections.defaultdict(lambda: collections.Counter())
    misses, cost, t0 = [], 0.0, time.time()

    for r in rows:
        src = r["path"].split("/")[1]
        drawing = os.path.splitext(os.path.basename(r["path"]))[0]
        t1 = time.time()
        if a.api:
            import urllib.request
            req = urllib.request.Request(
                a.api, method="POST",
                data=json.dumps({"question": r["question"], "drawing": drawing}).encode(),
                headers={"content-type": "application/json"})
            try:
                res = json.load(urllib.request.urlopen(req, timeout=30))
            except Exception as e:
                res = {"status": "refused", "reason": str(e)}
        else:
            res = A.ask(con, drawing, r["question"])
        ms = int((time.time() - t1) * 1000)
        cost += res.get("cost_usd", 0.0)

        got = res.get("answer") if res.get("status") in ("answered", "estimated") else None
        v = verdict(r["answer"], got, r["type"])
        stats[(src, r["type"])][v] += 1
        stats[("TOTAL", "")][v] += 1

        con.execute(
            """INSERT INTO eval_results
               (run_id, path, question, expected, got, verdict, q_type, source,
                route, cost_usd, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, r["path"], r["question"], str(r["answer"]), str(got), v,
             r["type"], src, res.get("route"), res.get("cost_usd", 0.0), ms))

        if v == "wrong" and len(misses) < a.show:
            misses.append(f"  {src:16s} {r['type']:6s} want {str(r['answer']):>9s} "
                          f"got {str(got):>9s}  {r['question'][:62]}")

    con.commit()

    print(f"\nrun {run_id} - {len(rows)} questions - {time.time()-t0:.0f}s "
          f"· ${cost:.4f}\n")
    print(f"{'source':18s}{'type':8s}{'right':>7s}{'wrong':>9s}"
          f"{'refused':>8s}{'skip':>7s}{'accuracy':>10s}")
    for (src, qt), c in sorted(stats.items()):
        n = c["correct"] + c["wrong"] + c["refused"]
        acc = f"{c['correct']*100/n:.0f}%" if n else "—"
        print(f"{src:18s}{qt:8s}{c['correct']:7d}{c['wrong']:9d}"
              f"{c['refused']:8d}{c['skipped']:7d}{acc:>10s}")

    tot = stats[("TOTAL", "")]
    n = tot["correct"] + tot["wrong"] + tot["refused"]
    if n:
        print(f"\nwrong-answer rate: {tot['wrong']*100/n:.1f}%  "
              f"(this is the number we quote)")
        print(f"coverage: answered {(tot['correct']+tot['wrong'])*100/n:.0f}%, "
              f"refused {tot['refused']*100/n:.0f}%")

    if misses:
        print("\nmisses:")
        print("\n".join(misses))
        print(f"\nall misses: sqlite3 {a.db} \"SELECT * FROM eval_results "
              f"WHERE run_id='{run_id}' AND verdict='wrong'\"")


if __name__ == "__main__":
    main()
