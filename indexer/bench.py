#!/usr/bin/env python3
"""Product-path benchmark: questions.jsonl through the running server, the way a judge uses it.

    python bench.py <dataset folder> [--server http://localhost:3000] [--source swiss_dwellings]
                    [--files 8] [--seed 1] [--out results.jsonl] [--show 15]

For every sampled file: upload it with its kind (what a judge does), then ask each of its questions.
Counts must match exactly; area, length and volume within 2%; text answers are written to --out for
reading by hand. A question that exists in several languages is one check; the language that is
asked is picked at random, so the model path is exercised in all of them.
Reports right / wrong / refused, share answered without a model, tokens, cost and latency.
"""
import argparse
import collections
import json
import os
import random
import re
import subprocess
import time
import urllib.request

KIND = {"swiss_dwellings": "dxf", "ifc_models": "ifc", "ifc_bench": "ifc",
        "opening_design": "pdf", "cubicasa5k": "png"}
PLAN_RE = re.compile(r"^(?P<model>.+?)_plan-\d+-(?P<floor>.+)$")
TOL = 0.02


def curl(*args, timeout=180):
    return subprocess.run(["curl", "-s", "-m", str(timeout), *args], capture_output=True, text=True).stdout


def upload(server, kind, path):
    out = curl(server + "/api/upload", "-F", f"kind={kind}", "-F", f"files=@{path}")
    try:
        job = json.loads(out)["jobId"]
    except Exception:
        return None, f"upload failed: {out[:120]}"
    body = curl("-N", f"{server}/api/jobs/{job}", timeout=240)
    last = [l for l in body.splitlines() if l.startswith("data:")]
    if not last:
        return None, "no job result"
    j = json.loads(last[-1][5:])
    if j.get("stage") != "ready" or not j.get("drawings"):
        return None, (j.get("log") or ["failed"])[-1].strip()[:160]
    return j["drawings"][0]["id"], None


def ask(server, uid, question):
    req = urllib.request.Request(
        server + "/api/ask", method="POST",
        data=json.dumps({"question": question, "drawing_id": uid}).encode(),
        headers={"content-type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))
    except Exception as e:
        return {"status": "refused", "reason": f"client: {e}"}


def verdict(expected, got, qtype):
    if got is None:
        return "refused"
    try:
        g = float(got)
        e = float(expected)
    except (TypeError, ValueError):
        return "wrong" if qtype != "text" else "text"
    if qtype == "count":
        return "right" if g == e else "wrong"
    if e == 0:
        return "right" if g == 0 else "wrong"
    return "right" if abs(g - e) <= TOL * abs(e) else "wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--server", default="http://localhost:3000")
    ap.add_argument("--source", default=None)
    ap.add_argument("--files", type=int, default=8, help="files sampled per source")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    rows = [json.loads(l) for l in open(os.path.join(a.root, "questions.jsonl"), encoding="utf-8") if l.strip()]
    if a.source:
        rows = [r for r in rows if f"/{a.source}/" in r["path"]]

    # group translations: same file, same answer, same type
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["path"], str(r["answer"]), r["type"])].append(r)

    # path -> file to upload (a plan PDF rendered from an IFC model is asked on the model)
    def target(path):
        src, name = path.split("/")[1], os.path.basename(path)
        stem = os.path.splitext(name)[0]
        d = os.path.join(a.root, "data", src)
        if src == "swiss_dwellings":
            return src, os.path.join(d, stem + ".dxf")
        if src in ("ifc_models",) and name.endswith((".pdf", ".svg")):
            m = PLAN_RE.match(stem)
            return src, os.path.join(d, (m.group("model") if m else stem) + ".ifc")
        if src == "opening_design":
            return src, os.path.join(a.root, path)
        return src, os.path.join(a.root, path)

    by_file = collections.defaultdict(list)
    for (path, _, _), g in groups.items():
        src, f = target(path)
        by_file[(src, f)].append(g)

    chosen = []
    for src in sorted({s for s, _ in by_file}):
        files = sorted(f for s, f in by_file if s == src)
        rnd.shuffle(files)
        chosen += [(src, f) for f in files[: a.files]]

    stats = collections.defaultdict(collections.Counter)
    money = collections.defaultdict(lambda: [0, 0, 0.0, 0])      # in, out, cost, calls
    lat = collections.defaultdict(list)
    misses, out = [], []
    t0 = time.time()

    for src, f in chosen:
        if not os.path.exists(f):
            stats[(src, "skipped")]["no source file"] += sum(len(g) for g in by_file[(src, f)])
            continue
        uid, err = upload(a.server, KIND[src], f)
        if not uid:
            for g in by_file[(src, f)]:
                stats[(src, g[0]["type"])]["refused"] += 1
            print(f"  ! {os.path.basename(f)}: {err}")
            continue
        for g in by_file[(src, f)]:
            r = rnd.choice(g)
            t1 = time.time()
            res = ask(a.server, uid, r["question"])
            ms = int((time.time() - t1) * 1000)
            ok = res.get("status") in ("answered", "estimated")
            got = res.get("answer") if ok else None
            v = verdict(r["answer"], got, r["type"])
            stats[(src, r["type"])][v] += 1
            route = res.get("route") or "?"
            stats[(src, "route")][route] += 1
            for u in res.get("usage") or []:
                m = money[u["model"]]
                m[0] += u["input_tokens"]; m[1] += u["output_tokens"]; m[2] += u["cost_usd"]; m[3] += 1
            lat[route].append(ms)
            rec = {"source": src, "file": os.path.basename(f), "type": r["type"], "q": r["question"],
                   "expected": r["answer"], "got": got, "status": res.get("status"), "verdict": v,
                   "route": route, "reason": res.get("reason"), "ms": ms, "cost": res.get("cost_usd", 0),
                   "fn": res.get("fn"), "args": res.get("args")}
            out.append(rec)
            if v == "wrong" and len(misses) < a.show:
                misses.append(f"  {src[:12]:12s} {r['type']:6s} want {str(r['answer'])[:9]:>9s} got {str(got)[:9]:>9s}"
                              f"  [{res.get('fn')}] {r['question'][:70]}")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            for rec in out:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{len(chosen)} files, {len(out)} questions, {time.time() - t0:.0f}s\n")
    print(f"{'source':17s}{'type':8s}{'right':>7s}{'wrong':>7s}{'refused':>9s}{'text':>6s}")
    tot = collections.Counter()
    for (src, qt), c in sorted(stats.items()):
        if qt in ("route", "skipped"):
            continue
        print(f"{src:17s}{qt:8s}{c['right']:7d}{c['wrong']:7d}{c['refused']:9d}{c['text']:6d}")
        tot.update(c)
    n = tot["right"] + tot["wrong"] + tot["refused"]
    if n:
        print(f"\nscored {n}: right {tot['right']}, wrong {tot['wrong']} ({tot['wrong'] * 100 / n:.1f}%), "
              f"refused {tot['refused']} ({tot['refused'] * 100 / n:.0f}%)   <- the number we quote is wrong")
    routes = collections.Counter()
    for (s, qt), c in stats.items():
        if qt == "route":
            routes.update(c)
    print("routes:", dict(routes))
    for route, ms in sorted(lat.items()):
        ms.sort()
        print(f"  {route:9s} n={len(ms):4d}  median {ms[len(ms) // 2]} ms  p95 {ms[int(len(ms) * .95)]} ms")
    for model, (i, o, c, calls) in money.items():
        print(f"model {model}: {calls} calls, {i} in / {o} out tokens, ${c:.4f}"
              f"  (${c / max(calls, 1):.5f} per call)")
    if misses:
        print("\nwrong answers:")
        print("\n".join(misses))


if __name__ == "__main__":
    main()
