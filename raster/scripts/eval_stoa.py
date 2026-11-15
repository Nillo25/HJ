"""Прогон набора Stoa (eval_set: images/ + questions.jsonl) через продукт как есть:
загрузка плана (infer.run: YOLO + разбор листа ИИ) -> вопрос в чат (правила, кэш, иначе Haiku) -> число.

    .venv/bin/python scripts/eval_stoa.py --set "<папка eval_set>" --out runs/eval_stoa/run1 [--workers 4]
    .venv/bin/python scripts/eval_stoa.py --set ... --out runs/eval_stoa/run1 --report   # только сводка

Считаются английские вопросы (финские дублируют их с тем же ответом). Сцены и ответы сохраняются в --out,
повторный запуск продолжает с места остановки. Точное совпадение числа — правило Stoa.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = str(ROOT / "runs/colab/v2_best.pt")
SPLITS = {}  # имя картинки -> train/val/test в нашем обучении YOLO (официальное разбиение CubiCasa)


def _our_split(name: str) -> str:
    k = name.replace("-", "__", 1)
    for sp in ("train", "val", "test"):
        if (ROOT / "data/yolo/images" / sp / k).exists():
            return sp
    return "none"


def _number(out: dict) -> float | None:
    """Ответ числом: результат последнего инструмента-подсчёта, иначе первое число в тексте ответа."""
    for c in reversed(out.get("calls") or []):
        a = c["result"].get("answer")
        if c["tool"] in ("count", "highlight") and isinstance(a, (int, float)):
            return a
    m = re.search(r"-?\d+(?:[.,]\d+)?", out.get("reply") or "")
    if m:
        return float(m.group().replace(",", "."))
    words = {"none": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for w in re.findall(r"[a-z]+", (out.get("reply") or "").lower()):
        if w in words:
            return words[w]
    return None


def run_plan(image: str, questions: list[dict], set_dir: str, out_dir: str) -> list[dict]:
    import server.app  # noqa: F401 — читает .env (ключ) и настраивает пути
    from fp import cache, infer
    from fp.chat import chat
    from fp.scene import Scene
    cache.configure(Path(out_dir) / "cache.sqlite")
    name = Path(image).name
    sdir = Path(out_dir) / "scenes" / Path(image).stem
    t0 = time.perf_counter()
    if (sdir / "scene.json").exists():
        sc, upload_s = Scene.load(sdir), 0.0
    else:
        sc = infer.run(Path(set_dir) / image, sdir, model=MODEL)
        upload_s = time.perf_counter() - t0
    rows = []
    for q in questions:
        try:
            out = chat(sc, q["question"], [])
            err = None
        except Exception as e:  # noqa: BLE001
            out, err = {"reply": "", "calls": []}, f"{type(e).__name__}: {e}"
        got = _number(out)
        rows.append({"image": name, "qtype": q["qtype"], "expected": q["answer"], "got": got,
                     "correct": got is not None and float(got) == float(q["answer"]),
                     "parsed_by": out.get("parsed_by"), "confidence": out.get("confidence"),
                     "cost_usd": out.get("cost_usd", 0.0), "latency_ms": out.get("latency_ms"),
                     "tools": ";".join(f"{c['tool']}{json.dumps(c['args'], ensure_ascii=False)}" for c in out.get("calls") or []),
                     "reply": (out.get("reply") or "")[:300], "error": err,
                     "cubicasa_test_split": q["cubicasa_test_split"], "our_split": _our_split(name),
                     "sheet_cost_usd": sc.meta.get("sheet_cost_usd", 0.0), "upload_s": round(upload_s, 1)})
    return rows


FIELDS = ["image", "qtype", "expected", "got", "correct", "parsed_by", "confidence", "cost_usd", "latency_ms",
          "tools", "reply", "error", "cubicasa_test_split", "our_split", "sheet_cost_usd", "upload_s"]


def report(out_dir: Path, set_dir: Path) -> str:
    rows = list(csv.DictReader(open(out_dir / "results.csv")))
    ours = {(r["image"], r["qtype"]): r for r in csv.DictReader(open(set_dir / "our_results.csv"))}
    tr = lambda r: r["correct"] == "True"
    test = lambda r: r["cubicasa_test_split"] == "True"
    lines = ["| Тип | n все / test | Мы: все | Мы: test | Гибрид: test | Opus: test | CubiCasa: test |",
             "|---|---|---|---|---|---|---|"]

    def pct(rs, f=tr):
        return f"{100 * sum(map(f, rs)) / len(rs):.1f}%" if rs else "—"

    def other(rs, col):
        o = [ours[(r["image"], r["qtype"])] for r in rs if (r["image"], r["qtype"]) in ours]
        return pct(o, lambda x: str(x[col]).strip() == str(x["expected"]).strip())
    by = defaultdict(list)
    for r in rows:
        by[r["qtype"]].append(r)
    for t in sorted(by, key=lambda t: -len(by[t])):
        rs = by[t]
        ts = [r for r in rs if test(r)]
        lines.append(f"| {t} | {len(rs)} / {len(ts)} | {pct(rs)} | {pct(ts)} | {other(ts, 'hybrid')} | "
                     f"{other(ts, 'opus_5_5')} | {other(ts, 'cubicasa')} |")
    ts = [r for r in rows if test(r)]
    lines.append(f"| **ВСЕГО** | {len(rows)} / {len(ts)} | **{pct(rows)}** | **{pct(ts)}** | {other(ts, 'hybrid')} | "
                 f"{other(ts, 'opus_5_5')} | {other(ts, 'cubicasa')} |")
    # честность: по нашему разбиению YOLO
    sp = defaultdict(list)
    for r in rows:
        sp[r["our_split"]].append(r)
    lines.append("\nПо разбиению нашего обучения YOLO: " +
                 ", ".join(f"{k} {pct(v)} ({len(v)} вопр.)" for k, v in sorted(sp.items())))
    # маршрут и цена
    pb = defaultdict(list)
    for r in rows:
        pb[r["parsed_by"]].append(r)
    lines.append("Кто разобрал вопрос: " + ", ".join(f"{k} {len(v)} (верно {pct(v)})" for k, v in sorted(pb.items())))
    plans = {r["image"]: float(r["sheet_cost_usd"] or 0) for r in rows}
    q_cost = sum(float(r["cost_usd"] or 0) for r in rows)
    lines.append(f"Цена: разбор листов ${sum(plans.values()):.4f} ({len(plans)} планов, "
                 f"${sum(plans.values()) / max(1, len(plans)):.4f}/план); вопросы ${q_cost:.4f} "
                 f"(${q_cost / max(1, len(rows)):.5f}/вопрос)")
    cf = defaultdict(list)
    for r in ts:
        cf[r["confidence"]].append(r)
    lines.append("Уверенность на test: " + ", ".join(f"{k}: {len(v)} отв., верно {pct(v)}" for k, v in sorted(cf.items(), key=str)))
    errs = [r for r in rows if r["error"]]
    if errs:
        lines.append(f"Ошибок выполнения: {len(errs)} (первая: {errs[0]['error']})")
    txt = "\n".join(lines)
    (out_dir / "summary.md").write_text(txt + "\n")
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="только первые N планов (проба)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    set_dir, out_dir = Path(a.set), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.report:
        print(report(out_dir, set_dir))
        return
    qs = defaultdict(list)
    for line in open(set_dir / "questions.jsonl"):
        q = json.loads(line)
        if q["question"].startswith("How"):
            qs[q["image"]].append(q)
    done = set()
    res_path = out_dir / "results.csv"
    if res_path.exists():
        done = {r["image"] for r in csv.DictReader(open(res_path))}
    todo = [im for im in sorted(qs) if Path(im).name not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"планов: {len(qs)}, готово: {len(done)}, осталось: {len(todo)}", flush=True)
    new = not res_path.exists()
    with open(res_path, "a", newline="") as f, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        futs = {ex.submit(run_plan, im, qs[im], str(set_dir), str(out_dir)): im for im in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            im = futs[fu]
            try:
                rows = fu.result()
            except Exception as e:  # noqa: BLE001 — план не разобрался: пропускаем, пишем в лог
                print(f"[{n}/{len(todo)}] {im}: ОШИБКА {type(e).__name__}: {e}", flush=True)
                continue
            w.writerows(rows)
            f.flush()
            ok = sum(r["correct"] for r in rows)
            print(f"[{n}/{len(todo)}] {Path(im).name}: {ok}/{len(rows)} верно, "
                  f"лист ${rows[0]['sheet_cost_usd']}, вопросы ${sum(r['cost_usd'] for r in rows):.4f}", flush=True)
    print(report(out_dir, set_dir))


if __name__ == "__main__":
    main()
