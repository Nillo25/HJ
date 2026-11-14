"""Склейка примитивов DXF в объекты + классификация сантехники.

Одна дверь или один прибор нарисованы десятком отрезков, дуг и окружностей.
Пока примитивы не склеены в объекты, считать и классифицировать нечего.
Это чинит разом два слабых места: двери (56%) и сантехнику (не различена).

Классификация — по габариту и по комнате, в которой объект лежит.
Пороги подобраны под европейские планы swiss_dwellings; на других
конвенциях черчения требуют перенастройки (говорим об этом вслух).
"""
import math

# ------------------------------------------------------------- геометрия

def bbox_of(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def bbox_gap(a, b) -> float:
    """Зазор между рамками. 0 = пересекаются или касаются."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def dims(bb):
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    lo, hi = min(w, h), max(w, h)
    return w, h, hi, lo, (hi / lo if lo > 1e-9 else 999.0)


# --------------------------------------------------------------- склейка

def cluster(prims, gap=0.05):
    """prims: [{'bbox': [...], ...}]. Объединяет всё, что ближе gap метров.

    Простое объединение множеств по парам. Примитивов на слое сотни,
    квадратичная сложность здесь не мешает.
    """
    n = len(prims)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    for i in range(n):
        for j in range(i + 1, n):
            if bbox_gap(prims[i]["bbox"], prims[j]["bbox"]) <= gap:
                union(i, j)

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(prims[i])

    out = []
    for members in groups.values():
        bb = members[0]["bbox"]
        for m in members[1:]:
            bb = bbox_union(bb, m["bbox"])
        out.append({"bbox": bb, "members": members, "n": len(members),
                    "has_circle": any(m.get("shape") == "circle" for m in members),
                    "has_arc": any(m.get("shape") == "arc" for m in members)})
    return out


# -------------------------------------------------- классификация приборов
# Реальные габариты, метры. Полосы специально широкие: лучше промахнуться
# в сторону "other", чем назвать ванну душем.

def classify_fixture(group, room_subtype: str | None) -> tuple[str | None, float]:
    """-> (subtype, confidence). None = не опознан, пойдёт как fixture/other."""
    _, _, long, short, aspect = dims(group["bbox"])
    area = long * short

    # ванна: длинная и узкая
    if 1.4 <= long <= 2.0 and 0.6 <= short <= 0.9:
        return "bathtub", 0.9

    # душевой поддон: почти квадрат, заметно меньше ванны
    if 0.7 <= long <= 1.2 and aspect <= 1.35:
        return "shower", 0.85

    # унитаз: вытянутый, маленький; на кухне не бывает
    if 0.5 <= long <= 0.85 and 0.3 <= short <= 0.5 and room_subtype != "kitchen":
        if room_subtype in ("wc", "bathroom", None):
            return "toilet", 0.85 if room_subtype else 0.6

    # раковина / мойка: маленькая, часто круглая
    if 0.3 <= long <= 0.9 and short <= 0.65:
        # контекст комнаты решает трудную пару унитаз / раковина
        if room_subtype == "kitchen":
            return "sink", 0.9
        if group["has_circle"]:
            return "sink", 0.8
        return "sink", 0.55

    return None, 0.3


# ------------------------------------------------------- двери из примитивов

def classify_door(group) -> tuple[str, float]:
    """Тип двери по склеенной группе.

    Наивное правило «одна дуга = одна дверь» давало 56%: у раздвижных дуги
    нет, у двустворчатых их две. После склейки считаем группы, а дуги внутри
    группы определяют подтип.
    """
    arcs = sum(1 for m in group["members"] if m.get("shape") == "arc")
    _, _, long, _, _ = dims(group["bbox"])
    if arcs >= 2:
        return "double", 0.9
    if arcs == 1:
        return "single", 0.9
    if long >= 0.6:
        return "sliding", 0.6      # полотно без дуги
    return "opening", 0.5          # проём без полотна — НЕ считается дверью


# ------------------------------------------------------------- диагностика

def histogram(groups, bins=(0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0)):
    """Распределение габаритов — смотреть ПЕРЕД подбором порогов.

    Если размеры лягут понятными полосами, пороги очевидны.
    Если размажутся — склейка работает плохо, чинить её, а не пороги.
    """
    rows = []
    for g in groups:
        _, _, long, short, aspect = dims(g["bbox"])
        rows.append((round(long, 2), round(short, 2), round(aspect, 2), g["n"]))
    rows.sort(reverse=True)
    counts = {b: 0 for b in bins}
    for long, *_ in rows:
        for b in bins:
            if long <= b:
                counts[b] += 1
                break
    return rows, counts
