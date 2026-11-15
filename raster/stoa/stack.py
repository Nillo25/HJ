"""P1: stacking / calibration of per-question answers from several sources.

Sources (counts per plan x qtype):
  cc      - CubiCasa x1 (bench_cubicasa.csv)
  haiku   - Claude Haiku 5.5 batch (bench_claude_claude-haiku-5-5.csv), all plans
  opus    - Claude Opus 5.5 batch (bench_claude_claude-opus-5-5.csv), 80 plans only
  f_*     - P2 multi-scale fusion counts (fusion.py), plans where the scale caches exist

Candidate models per qtype (all fit ONLY on plans with in_test=False):
  pick:s          answer = source s
  pick+b:s        answer = source s + integer offset b (bias correction)
  agree:s1>s2     answer = s1 if s1==s2 else s2's ... (if agree -> common value, else fallback source)
  median:a,b,c    median of three sources
  lin:[...]       round(least-squares linear combination of sources + const)
Model selection: grouped 5-fold CV on train plans; the winner is refit on all train plans and scored on in_test.

python -m stack [--sources cc,haiku] [--opus]
"""
import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config

QTYPES = ["windows", "doors", "toilets", "sinks", "showers", "saunas", "kitchens", "bedrooms", "bathrooms",
          "wc_rooms", "balconies", "floors"]
BENCH = {"cc": "bench_cubicasa.csv", "haiku": "bench_claude_claude-haiku-5-5.csv",
         "opus": "bench_claude_claude-opus-5-5.csv", "hybrid": "bench_hybrid.csv"}
FUSION = {  # name -> fusion.fused kwargs
    "f_x15": dict(mode="single", views=["cubicasa_s1.5"]),
    "f_x2": dict(mode="single", views=["cubicasa_s2"]),
    "f_union": dict(mode="vote", k=1, thr=0.4),
    "f_vote2": dict(mode="vote", k=2, thr=0.4),
    "f_pmean": dict(mode="pmean", pthr=0.3),
    "f_adapt": dict(mode="adapt", ppm_thr=30, hi_view="cubicasa_s1.5"),
}
FUSION_Q = {"windows", "doors", "toilets", "sinks"}


def load_rows():
    """{(stem,qtype): {"exp":, "test":, "subtype":, "cc":, "haiku":, ...}}"""
    D = {}
    for src, f in BENCH.items():
        p = config.REPORTS_DIR / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            k = (Path(r["path"]).stem, r["qtype"])
            d = D.setdefault(k, {"exp": int(r["expected"]), "test": r["in_test"] == "True", "subtype": r["subtype"]})
            try:
                d[src] = int(float(r["got"]))
            except ValueError:
                pass
            if src == "cc":
                d["cc_conf"] = r["confidence"]
    return D


def add_fusion(D, cache_path=None):
    from . import fusion
    cache_path = cache_path or config.REPORTS_DIR / "fusion_counts.csv"
    have = {}
    if cache_path.exists():
        for r in csv.DictReader(open(cache_path)):
            have[(r["stem"], r["qtype"], r["src"])] = int(r["n"])
    new = []
    for (stem, q), d in D.items():
        if q not in FUSION_Q:
            continue
        ok = all(fusion.has_view(stem, v) for v in fusion.VIEWS)
        for name, kw in FUSION.items():
            key = (stem, q, name)
            if key not in have and ok:
                have[key] = len(fusion.fused(stem, q, **kw))
                new.append(key)
            if key in have:
                d[name] = have[key]
    if new:
        with open(cache_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stem", "qtype", "src", "n"])
            for (s, q, n), v in sorted(have.items()):
                w.writerow([s, q, n, v])
    return D


# ------------------------------------------------------------------------------------------- candidate models
class Pick:
    def __init__(self, s, b=0):
        self.s, self.b, self.name = s, b, f"{s}{b:+d}" if b else s

    def need(self):
        return [self.s]

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[self.s] + self.b


class PickBias(Pick):
    def __init__(self, s):
        super().__init__(s)
        self.name = f"{s}+b"

    def fit(self, X, y):
        bs = range(-3, 4)
        self.b = max(bs, key=lambda b: (np.mean(X[self.s] + b == y), -abs(b)))
        self.name = f"{self.s}{self.b:+d}" if self.b else self.s
        return self


class Agree:
    """if a == b -> a, else fallback source c"""
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c
        self.name = f"agree({a},{b})else {c}"

    def need(self):
        return list({self.a, self.b, self.c})

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.where(X[self.a] == X[self.b], X[self.a], X[self.c])


class Median:
    def __init__(self, srcs):
        self.srcs = srcs
        self.name = "median(" + ",".join(srcs) + ")"

    def need(self):
        return list(self.srcs)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.round(np.median(np.stack([X[s] for s in self.srcs]), 0) + 1e-9)


class Lin:
    """round(w.x + c), ridge least squares"""
    def __init__(self, srcs, lam=1.0):
        self.srcs, self.lam = srcs, lam
        self.name = "lin(" + ",".join(srcs) + ")"

    def need(self):
        return list(self.srcs)

    def _A(self, X):
        return np.stack([X[s] for s in self.srcs] + [np.ones(len(X[self.srcs[0]]))], 1).astype(float)

    def fit(self, X, y):
        A = self._A(X)
        R = self.lam * np.eye(A.shape[1])
        R[-1, -1] = 0
        self.w = np.linalg.solve(A.T @ A + R, A.T @ y)
        return self

    def predict(self, X):
        return np.round(self._A(X) @ self.w)


def candidates(srcs):
    out = []
    for s in srcs:
        out += [Pick(s), PickBias(s)]
    for a, b in itertools.combinations(srcs, 2):
        for c in srcs:
            out.append(Agree(a, b, c))
        out.append(Lin([a, b]))
    for tri in itertools.combinations(srcs, 3):
        out += [Median(list(tri)), Lin(list(tri))]
    if len(srcs) > 3:
        out.append(Lin(list(srcs), lam=3.0))
    return out


# ------------------------------------------------------------------------------------------- CV / selection
def to_X(items, srcs):
    return {s: np.array([d[s] for d in items], float) for s in srcs}


def cv_score(model, items, srcs, folds=5, seed=0):
    stems = sorted({d["stem"] for d in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)
    fold_of = {s: i % folds for i, s in enumerate(stems)}
    hits = []
    for f in range(folds):
        tr = [d for d in items if fold_of[d["stem"]] != f]
        te = [d for d in items if fold_of[d["stem"]] == f]
        if not te or not tr:
            continue
        m = model.fit(to_X(tr, srcs), np.array([d["exp"] for d in tr], float))
        hits += list(m.predict(to_X(te, srcs)) == np.array([d["exp"] for d in te]))
    return float(np.mean(hits)) if hits else 0.0


def run(srcs, qtypes=QTYPES, verbose=True, only_plans=None):
    D = load_rows()
    if any(s.startswith("f_") for s in srcs):
        D = add_fusion(D)
    out = {}
    for q in qtypes:
        qs = [s for s in srcs if not (s.startswith("f_") and q not in FUSION_Q)]
        items = [dict(d, stem=k[0]) for k, d in D.items() if k[1] == q and all(s in d for s in qs)]
        if only_plans is not None:
            items = [d for d in items if d["stem"] in only_plans]
        tr = [d for d in items if not d["test"]]
        te = [d for d in items if d["test"]]
        if len(tr) < 10 or not te:
            continue
        cands = candidates(qs)
        scored = sorted(((cv_score(m, tr, qs), i) for i, m in enumerate(cands)), key=lambda t: -t[0])
        cv, bi = scored[0]
        best = cands[bi].fit(to_X(tr, qs), np.array([d["exp"] for d in tr], float))
        yte = np.array([d["exp"] for d in te])
        Xte = to_X(te, qs)
        res = {"model": best.name, "cv": cv, "n_train": len(tr), "n_test": len(te),
               "test_stack": float(np.mean(best.predict(Xte) == yte))}
        for s in qs:
            res[f"test_{s}"] = float(np.mean(Xte[s] == yte))
            res[f"train_{s}"] = float(np.mean(to_X(tr, qs)[s] == np.array([d["exp"] for d in tr])))
        res["train_stack_cv"] = cv
        out[q] = res
    if verbose:
        print(f"\nsources: {srcs}")
        print(f"{'qtype':10s} {'chosen model':34s} {'CV train':>8s} " + " ".join(f"{'T:'+s:>8s}" for s in srcs) + f" {'T:stack':>8s}  n_tr/n_te")
        tot = defaultdict(float)
        for q, r in out.items():
            cols = " ".join(f"{r.get('test_'+s, float('nan')):8.0%}" for s in srcs)
            print(f"{q:10s} {r['model'][:34]:34s} {r['cv']:8.0%} {cols} {r['test_stack']:8.0%}  {r['n_train']}/{r['n_test']}")
            for s in srcs + ["stack"]:
                if f"test_{s}" in r:
                    tot[s] += r[f"test_{s}"] * r["n_test"]
            tot["n"] += r["n_test"]
        n = tot.pop("n")
        print(f"{'ALL (test)':10s} {'':34s} {'':8s} " + " ".join(f"{tot.get(s, 0)/n:8.0%}" for s in srcs) + f" {tot['stack']/n:8.0%}  n_te={int(n)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="cc,haiku")
    a = ap.parse_args()
    run(a.sources.split(","))


if __name__ == "__main__":
    main()
