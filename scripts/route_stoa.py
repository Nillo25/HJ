"""Выбор источника ответа по типу вопроса (YOLO или опись плана) на планах, которых нет в test CubiCasa,
и проверка разбора вопросов. Пишет fp/route.json. Test не смотрим — по нему только отчёт (eval_stoa.py).

    .venv/bin/python scripts/route_stoa.py --set "<eval_set>" --scenes runs/eval_stoa/run1/scenes
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fp.chat import parse_rule  # noqa: E402
from fp.query import INV_KEY, VIRTUAL, _count_yolo  # noqa: E402
from fp.scene import Scene  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True)
ap.add_argument("--scenes", required=True)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()
acc = defaultdict(lambda: Counter())
parsed = defaultdict(Counter)
for line in open(Path(a.set) / "questions.jsonl"):
    q = json.loads(line)
    if not q["question"].startswith("How") or q["cubicasa_test_split"]:
        continue
    rule = parse_rule(q["question"])
    cls = rule[1].get("cls") if rule else None
    parsed[q["qtype"]][f"{rule[0]}:{cls}" if rule else "LLM"] += 1
    if cls not in INV_KEY:
        continue
    sc = Scene.load(Path(a.scenes) / Path(q["image"]).stem)
    inv = sc.meta["inventory"]
    v = inv[INV_KEY[cls]]
    c = acc[cls]
    c["n"] += 1
    c["inv"] += v == q["answer"]
    if cls not in VIRTUAL:
        y = _count_yolo(sc, cls).answer
        c["yolo"] += y == q["answer"]
        c["agree"] += y == v
        c["agree_ok"] += y == v == q["answer"]
print("разбор вопросов (train+val):")
for t, c in parsed.items():
    print(f"  {t}: {dict(c)}")
route = {}
print(f"\n{'класс':16} {'n':>3} {'YOLO':>6} {'опись':>6}  совпали (верно)")
for cls, c in sorted(acc.items()):
    n = c["n"]
    y = c["yolo"] / n if cls not in VIRTUAL else None
    i = c["inv"] / n
    route[cls] = "inventory" if y is None or i > y else "yolo"
    ag = f"{c['agree']}/{n} ({c['agree_ok']}/{c['agree']} верно)" if c["agree"] else "—"
    print(f"{cls:16} {n:>3} {('%.0f%%' % (100 * y)) if y is not None else '—':>6} {100 * i:>5.0f}%  {ag}  -> {route[cls]}")
if a.write:
    (ROOT / "fp/route.json").write_text(json.dumps(route, indent=1, ensure_ascii=False) + "\n")
    print("\nзаписано fp/route.json")
