"""Лист трубопроводов: длина труб по диаметрам и арматура — из подписей листа и линий труб.

1. ИИ один раз читает подписи (fp.sheet.read_piping): диаметры с точкой выноски, марки арматуры, приборов,
   опор и оборудования.
2. Код находит участки труб (fp.pipes_cv.find_pipes — двойные линии; малые трубы — одиночные линии).
3. От точки выноски каждой подписи диаметра прослеживается труба: участок под выноской и продолжающие его
   участки той же ширины (соосные через фитинги и арматуру, и через повороты). Длина трубы данного диаметра =
   сумма участков, привязанных к его подписям. Участки без подписи учитываются отдельно (диаметр по ширине).
4. Арматура и приборы — по маркам: CV-1…5 -> «CV» × 5 и т. д.
"""
from __future__ import annotations

import re
from collections import defaultdict

import cv2
import numpy as np

from .pipes_cv import _lines, find_pipes

GAP_M = 0.8           # разрыв на одной оси (арматура, фитинг, подпись) — та же труба
TURN_M = 0.5          # поворот: конец участка рядом с концом/телом перпендикулярного участка
WIDTH_TOL = 0.35      # ширина «той же» трубы ±35%
SINGLE_GAP_M = 0.3    # малая труба одной линией: разрыв на той же оси не больше 0.3 м


def _seg_end_pts(s):
    if s["axis"] == "h":
        return [(s["a0"], s["c"]), (s["a1"], s["c"])]
    return [(s["c"], s["a0"]), (s["c"], s["a1"])]


def _dist_to_seg(s, x, y) -> float:
    if s["axis"] == "h":
        dx = max(s["a0"] - x, 0, x - s["a1"])
        return float(np.hypot(dx, abs(y - s["c"]) - min(abs(y - s["c"]), s["width_px"] / 2)))
    dy = max(s["a0"] - y, 0, y - s["a1"])
    return float(np.hypot(abs(x - s["c"]) - min(abs(x - s["c"]), s["width_px"] / 2), dy))


def _connected(a, b, ppm) -> bool:
    if abs(a["width_px"] - b["width_px"]) > WIDTH_TOL * max(a["width_px"], b["width_px"]):
        return False
    single = a.get("single") or b.get("single")
    if a["axis"] == b["axis"]:
        if abs(a["c"] - b["c"]) > (2 if single else max(3, 0.25 * a["width_px"])):
            return False
        gap = max(a["a0"], b["a0"]) - min(a["a1"], b["a1"])
        return gap <= (SINGLE_GAP_M if single else GAP_M) * ppm
    if single:
        return False      # одиночную линию через повороты не ведём: иначе уходит по стенам, рамкам и выноскам
    # поворот: конец одного у конца или тела другого
    return any(_dist_to_seg(b, x, y) <= TURN_M * ppm for x, y in _seg_end_pts(a))


def _single_lines(gray, ppm, region):
    """Малые трубы одной линией: длинные осевые линии, не входящие в двойные пары."""
    dark = (gray < 200).astype(np.uint8)
    if region:
        x0, y0, x1, y1 = (int(v) for v in region)
        m = np.zeros_like(dark)
        m[max(0, y0):y1, max(0, x0):x1] = 1
        dark &= m
    L = max(8, int(0.3 * ppm))
    out = []
    for axis in (0, 1):
        lines = _lines(dark, axis, L)
        for k, (c, a0, a1) in enumerate(lines):
            # малая труба — одиночная линия; линия с параллельной соседкой вплотную (1-5 px) по большей части
            # длины — стена, рамка, кромка листа (наружная стена M-101A дала «2"» на 16 м)
            twin = any(0 < abs(c2 - c) <= 5 and min(a1, b1) - max(a0, b0) > 0.5 * (a1 - a0)
                       for c2, b0, b1 in lines[max(0, k - 8):k + 8])
            if twin:
                continue
            out.append({"axis": "h" if axis == 0 else "v", "c": round(c, 1), "a0": int(a0), "a1": int(a1),
                        "width_px": 1.0, "length_m": round((a1 - a0) / ppm, 2), "nps": None,
                        "bbox": [a0, c - 1, a1, c + 1] if axis == 0 else [c - 1, a0, c + 1, a1], "single": True})
    return out


def _not_outline(singles, segs, rooms):
    """Одиночные линии, которые не кромка двойной трубы и не граница помещения (стена)."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    walls = unary_union([r.shape().boundary for r in rooms]) if rooms else None
    out = []
    for s in singles:
        edge = False
        for d in segs:
            if d["axis"] != s["axis"]:
                continue
            for e in (d["c"] - d["width_px"] / 2, d["c"] + d["width_px"] / 2):
                if abs(s["c"] - e) <= 2 and min(s["a1"], d["a1"]) - max(s["a0"], d["a0"]) > 0.5 * (s["a1"] - s["a0"]):
                    edge = True
        if edge:
            continue
        if walls is not None:
            ln = LineString(_seg_end_pts(s))
            if ln.buffer(3).intersection(walls).length > 0.5 * ln.length:
                continue
        out.append(s)
    return out


def _not_walls(segs, rooms):
    """Двойная пара вдоль границы помещения — это стена из двух тонких линий, а не труба."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    if not rooms:
        return segs
    walls = unary_union([r.shape().boundary for r in rooms])
    out = []
    for s in segs:
        ln = LineString(_seg_end_pts(s))
        if ln.buffer(s["width_px"] / 2 + 4).intersection(walls).length > 0.5 * ln.length:
            continue
        out.append(s)
    return out


def build(img: np.ndarray, sc, items: list[dict]) -> dict:
    """Трубы по подписям + арматура по маркам. items — из fp.sheet.read_piping."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ppm = sc.meta["px_per_m"]
    region = (sc.meta.get("sheet") or {}).get("drawing_bbox")
    segs = _not_walls(find_pipes(gray, ppm, region), sc.rooms)
    singles = _not_outline(_single_lines(gray, ppm, region), segs, sc.rooms)
    for i, s in enumerate(segs):
        s["id"] = f"pd{i + 1}"
    for i, s in enumerate(singles):
        s["id"] = f"pl{i + 1}"
    size_labels = [it for it in items if it["kind"] == "pipe_size" and it.get("size_in")]
    owner: dict = {}                      # id участка -> (размер, подпись)
    runs = []
    # сначала подписи, чья выноска ближе всего к трубе: спорные участки достаются более уверенной подписи
    cand = []
    for k, lb in enumerate(size_labels):
        tx, ty = lb["target_x"], lb["target_y"]
        dbl = min(segs, key=lambda s: _dist_to_seg(s, tx, ty), default=None)
        d_dbl = _dist_to_seg(dbl, tx, ty) if dbl else 1e9
        sgl = min(singles, key=lambda s: _dist_to_seg(s, tx, ty), default=None)
        d_sgl = _dist_to_seg(sgl, tx, ty) if sgl else 1e9
        # малые трубы (≤ 2") рисуют одной линией; крупные — двойной
        if lb["size_in"] <= 2 and d_sgl <= 0.15 * ppm:
            cand.append((d_sgl, k, sgl, singles))
        elif d_dbl <= 0.3 * ppm:
            cand.append((d_dbl, k, dbl, segs))
        elif d_sgl <= 0.15 * ppm:
            cand.append((d_sgl, k, sgl, singles))
    for _, k, seed, pool in sorted(cand, key=lambda t: t[0]):
        lb = size_labels[k]
        if seed["id"] in owner:
            runs.append({"label": lb["text"], "size_in": lb["size_in"], "ids": [], "length_m": 0.0,
                         "note": f"труба уже привязана к подписи {owner[seed['id']][1]}"})
            continue
        run, todo = [seed], [seed]
        owner[seed["id"]] = (lb["size_in"], lb["text"])
        while todo:
            a = todo.pop()
            for b in pool:
                if b["id"] not in owner and _connected(a, b, ppm):
                    owner[b["id"]] = (lb["size_in"], lb["text"])
                    run.append(b)
                    todo.append(b)
        runs.append({"label": lb["text"], "size_in": lb["size_in"], "ids": [s["id"] for s in run],
                     "length_m": round(sum(s["length_m"] for s in run), 2), "x": lb["x"], "y": lb["y"]})
    by_size: dict = defaultdict(float)
    for r in runs:
        by_size[r["size_in"]] += r["length_m"]
    unl = [s for s in segs if s["id"] not in owner]
    unl_by: dict = defaultdict(float)
    for s in unl:
        unl_by[s["nps"]] += s["length_m"]
    # арматура и приборы по маркам: префикс до цифр («FSCV-3» -> FSCV, «PIT 4» -> PIT)
    # группируем по префиксу марки, а не по типу: ИИ в разных фрагментах относит одну марку к разным типам
    tags: dict = defaultdict(list)
    for it in items:
        if it["kind"] in ("valve", "instrument", "support", "equipment"):
            txt = it["text"].upper().strip()
            # марка: 2-5 букв, по желанию номер (CV-3, PIT 4, DPT-1 & 2, FSI-5) — не ссылки на листы (D1 M-301A),
            # не слова из примечаний (NON-SURGE…), не названия (JOCKEY PUMP)
            m = re.fullmatch(r"([A-Z/]{2,5})(?:[- ]?\d+[A-Z]?(?:\s*&\s*\d+)?)?(?:\s*\(TYP\.?\))?", txt)
            if m:
                tags[m.group(1)].append(it)
    fittings = []
    for t, v in sorted(tags.items()):
        kinds = defaultdict(int)
        for i in v:
            kinds[i["kind"]] += 1
        numbered = [i for i in v if re.search(r"\d", i["text"])]
        # марки с номерами (CV-1…CV-5) — по уникальным номерам; без номеров (PG, MAV) — по числу подписей;
        # опоры PS-n — по числу выносок (одна марка ставится на несколько одинаковых опор)
        kind = max(kinds, key=kinds.get)
        n = len(v) if (kind == "support" or not numbered) else len({i["text"] for i in numbered})
        fittings.append({"kind": kind, "tag": t, "count": n, "texts": sorted({i["text"] for i in v}),
                         "points": [[i["target_x"], i["target_y"]] for i in v]})
    used = {s["id"]: s for s in segs + singles if s["id"] in owner}
    return {"by_size_m": {str(k): round(v, 2) for k, v in sorted(by_size.items())},
            "unlabeled_by_width_m": {str(k): round(v, 2) for k, v in sorted(unl_by.items(), key=lambda t: t[0] or 0)},
            "runs": runs, "fittings": fittings,
            "segments": [{**{k: s[k] for k in ("id", "axis", "bbox", "length_m", "width_px")},
                          "size_in": owner[s["id"]][0], "label": owner[s["id"]][1]} for s in used.values()],
            "labels_read": len(size_labels)}
