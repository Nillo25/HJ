"""Какие id/class реально встречаются в model.svg датасета — для проверки маппинга в fp/classes.py.

    python scripts/inspect_cubicasa.py data/cubicasa5k --limit 500
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fp.cubicasa_svg import parse_cubicasa_svg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    raw, mapped, unknown, n = Counter(), Counter(), Counter(), 0
    for svg in sorted(a.src.rglob("model.svg")):
        root = etree.parse(str(svg)).getroot()
        for el in root.iter():
            if isinstance(el.tag, str) and el.tag.endswith("g"):
                cls = (el.get("class") or "").strip()
                key = " ".join(cls.split(" ")[:2]) if cls else f"#id={el.get('id')}"
                raw[key] += 1
        p = parse_cubicasa_svg(svg)
        mapped.update(c for c, _ in p.objects)
        unknown.update(p.unknown)
        n += 1
        if a.limit and n >= a.limit:
            break
    print(f"файлов: {n}\n\n== сырые class (топ-80) ==")
    for k, v in raw.most_common(80):
        print(f"{v:7d}  {k}")
    print("\n== после маппинга ==")
    for k, v in mapped.most_common():
        print(f"{v:7d}  {k}")
    print("\n== не смаппилось ==")
    for k, v in unknown.most_common(40):
        print(f"{v:7d}  {k}")


if __name__ == "__main__":
    main()
