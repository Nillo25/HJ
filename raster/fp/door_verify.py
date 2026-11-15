"""Двери на чертёжных листах: кандидаты с запасом -> проверка ИИ по вырезкам.

Кандидаты: двери YOLO по тайлам с низким порогом (CAND_CONF) + ВСЕ дуги открывания (fp/doors_cv, без фильтра
«рядом со стеной» — стены модели на CAD-листах ненадёжны). ИИ получает вырезку вокруг каждого кандидата
(красная рамка в центре) и отвечает: дверь с полотном или нет (пустой проём, лестница, оборудование, выносная
линия — не дверь), одна или две створки. Все вырезки листа — одним-двумя вызовами (BATCH на вызов).

Замер (eval/doors_gt.json: M-101A 6 дверей, S-100F 0, фрагмент школы 26; scripts/exp_doors.py):
  проход по целому листу (как было):  TP 10, лишних 5, пропущено 22
  тайлы + проверка ИИ (Opus):         TP 32, лишних 1, пропущено 0   ($0.11 на 3 листа)
  те же кандидаты, Haiku (14.11.2026): TP 32, лишних 0, пропущено 0 в двух прогонах подряд, $0.01 на 3 листа
                                       (Opus в продукте ~$0.30) — по умолчанию Haiku; Opus: FP_DOOR_VERIFY_MODEL
"""
from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np

VERIFY_MODEL = os.environ.get("FP_DOOR_VERIFY_MODEL", "claude-haiku-5-5")
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
CAND_CONF = 0.15
ARC_WORK_PPM = 50.0  # дуги ищутся не крупнее 50 px/м: на школе 14000 px — 28 с -> 5 с, покрытие дверей 21/26 -> 22/26
WORKERS = 4          # пачки вырезок проверяются параллельно
DEDUP_M = 0.8
CROP_M = 3.0
CROP_PX = 320
BATCH = 16
SCHEMA = {"type": "object", "properties": {"crops": {"type": "array", "items": {"type": "object", "properties": {
    "crop": {"type": "integer"}, "door": {"type": "boolean"}, "leaves": {"type": "integer"}},
    "required": ["crop", "door", "leaves"], "additionalProperties": False}}},
    "required": ["crops"], "additionalProperties": False}
PROMPT = ("Each image is a crop of an architectural drawing sheet (top view). A red box marks a candidate found by a "
          "detector, at the centre of the crop. For EVERY crop answer: is there a DOOR at the red box — a doorway in a "
          "wall fitted with a door leaf. Doors are drawn as: a swing arc with a leaf line; a leaf drawn as a straight "
          "line at an angle to the wall WITHOUT an arc; a folding / bi-fold door drawn as a zigzag or a pair of short "
          "angled lines (like > or ^) across the opening; a double door (two arcs or two angled leaves); a sliding leaf "
          "parallel to the wall. NOT doors: an open doorway or gap in a wall with no leaf at all, stairs, windows "
          "(thin parallel glazing lines in the wall), equipment/cabinet/tank doors or clearances, pipes, leader lines, "
          "text, valves, dimension marks, furniture. `leaves`: 1 for a single door, 2 for a double door, 0 if not a door.")


def candidates(full: np.ndarray, ppm_full: float, model: str, yolo: list[dict] | None = None,
               region=None) -> list[dict]:
    """Кандидаты в пикселях full: {"c": [x, y], "box": [x0, y0, x1, y1], "src", "conf"}.
    yolo — двери тайлового прохода (fp.tiles.retile), чтобы не гонять YOLO второй раз; region — рамка чертежа
    в пикселях full: дуги ищутся только там (на листе 14000 px поиск по всему листу — минуты)."""
    from .doors_cv import find_door_arcs
    from .infer import THRESHOLDS
    from .tiles import detect_tiles
    th = json.loads(json.dumps(THRESHOLDS))
    th["door"]["conf"] = CAND_CONF
    if yolo is None:
        yolo = [{"c": p.mean(0).tolist(), "box": [*p.min(0), *p.max(0)], "src": "yolo", "conf": float(conf)}
                for cls, p, conf in detect_tiles(full, ppm_full, model, thresholds=th) if cls == "door"]
    out = [dict(c) for c in yolo]
    gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY) if full.ndim == 3 else full
    ox = oy = 0
    if region:
        m = int(1.5 * ppm_full)
        ox, oy = max(0, int(region[0]) - m), max(0, int(region[1]) - m)
        gray = gray[oy:int(region[3]) + m, ox:int(region[2]) + m]
    s = min(1.0, ARC_WORK_PPM / ppm_full)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    for a in find_door_arcs(g, ppm_full * s, None, None):
        b = np.array(a["bbox"], dtype=float) / s + (ox, oy, ox, oy)
        out.append({"c": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], "box": b.tolist(), "src": "arc", "conf": a["score"] / 2})
    keep: list[dict] = []
    for c in sorted(out, key=lambda d: -d["conf"]):
        near = [k for k in keep if np.hypot(c["c"][0] - k["c"][0], c["c"][1] - k["c"][1]) <= DEDUP_M * ppm_full]
        if near:
            near[0]["src"] = "+".join(sorted(set(near[0]["src"].split("+")) | {c["src"]}))
        else:
            keep.append(dict(c))
    return keep


def _crop_png(gray: np.ndarray, cand: dict, ppm: float) -> bytes:
    half = CROP_M * ppm / 2
    cx, cy = cand["c"]
    x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    H, W = gray.shape
    pad = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    im = cv2.cvtColor(cv2.resize(pad, (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    k = CROP_PX / (x1 - x0)
    b = cand["box"]
    cv2.rectangle(im, (int((b[0] - x0) * k) - 3, int((b[1] - y0) * k) - 3),
                  (int((b[2] - x0) * k) + 3, int((b[3] - y0) * k) + 3), (0, 0, 255), 2)
    return cv2.imencode(".png", im)[1].tobytes()


def verify(full: np.ndarray, cands: list[dict], ppm_full: float) -> tuple[list[dict], float]:
    """Помечает кандидатов полями door/leaves. Возвращает (кандидаты, цена $)."""
    import anthropic
    gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY) if full.ndim == 3 else full
    client = anthropic.Anthropic()
    pin, pout = PRICES.get(VERIFY_MODEL, (4.0, 20.0))
    cost = 0.0
    def run(part):
        content = []
        for n, c in enumerate(part, 1):
            content += [{"type": "text", "text": f"Crop {n}:"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.standard_b64encode(_crop_png(gray, c, ppm_full)).decode()}}]
        content.append({"type": "text", "text": PROMPT})
        kw = dict(model=VERIFY_MODEL, max_tokens=4000, messages=[{"role": "user", "content": content}],
                  output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
        if VERIFY_MODEL.startswith("claude-haiku"):
            r = client.messages.create(**kw)
        else:  # отказ классификатора -> серверный fallback (как в fp/inventory.py)
            r = client.beta.messages.create(**kw, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        c_usd = (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
        if r.stop_reason == "refusal":
            for c in part:
                c["door"], c["leaves"], c["unverified"] = True, 1, True  # не знаем — оставляем, уверенность ниже
            return c_usd
        ans = {a["crop"]: a for a in json.loads(next(b.text for b in r.content if b.type == "text"))["crops"]}
        for n, c in enumerate(part, 1):
            a = ans.get(n, {"door": False, "leaves": 0})
            c["door"], c["leaves"] = bool(a["door"]), int(a["leaves"])
        return c_usd

    from concurrent.futures import ThreadPoolExecutor
    parts = [cands[i:i + BATCH] for i in range(0, len(cands), BATCH)]
    with ThreadPoolExecutor(WORKERS) as ex:
        cost = sum(ex.map(run, parts))
    return cands, cost


def apply(sc, full: np.ndarray, f: float, model: str) -> dict:
    """Двери сцены = проверенные кандидаты (координаты сцены = full * f)."""
    from .scene import Element
    ppm_full = sc.meta["px_per_m"] / f
    bb = (sc.meta.get("sheet") or {}).get("drawing_bbox")
    region = [v / f for v in bb] if bb and len(bb) == 4 else None
    cands = candidates(full, ppm_full, model, yolo=getattr(sc, "door_candidates", None), region=region)
    cands, cost = verify(full, cands, ppm_full)
    sc.objects = [o for o in sc.objects if o.cls != "door"]
    n = 0
    for c in cands:
        if not c["door"]:
            continue
        n += 1
        x0, y0, x1, y1 = (v * f for v in c["box"])
        conf = 0.6 if c.get("unverified") else (0.9 if "yolo" in c["src"] else 0.8)
        sc.objects.append(Element(f"dv{n}", "door", [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], conf,
                                  source="verified:" + c["src"]))
    sc.assign_rooms()
    return {"done": True, "model": VERIFY_MODEL, "candidates": len(cands), "doors": n,
            "rejected": len(cands) - n, "double": sum(1 for c in cands if c.get("door") and c.get("leaves") == 2),
            "cost_usd": round(cost, 6)}
