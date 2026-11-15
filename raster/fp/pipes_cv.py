"""Трубы на листе трубопроводов: двойные линии -> участки с осью, длиной и диаметром.

На механических листах (M-101A, масштаб 1/4" = 1'-0") трубы от ~3" рисуют двумя параллельными линиями
на расстоянии наружного диаметра трубы в масштабе листа, малые (2" и меньше) — одной линией.
1. Длинные прямые линии (горизонтальные и вертикальные) — открытием маски тёмного длинным ядром.
2. Пара параллельных линий на расстоянии 0.06–0.5 м, перекрывающихся по длине, с почти пустым промежутком
   (внутри трубы белое) — участок трубы. Ширина пары = наружный диаметр -> ближайший стандартный размер.
3. Диаметр уточняется подписями («8"», «16" SUCTION HEADER») — fp.piping.
Длина участка = перекрытие пары; фитинги и арматура рвут линии, поэтому длины по участкам складываются.
"""
from __future__ import annotations

import cv2
import numpy as np

# наружный диаметр (дюймы) -> номинальный размер NPS
NPS_OD = {1.5: 1.9, 2: 2.375, 2.5: 2.875, 3: 3.5, 4: 4.5, 6: 6.625, 8: 8.625, 10: 10.75, 12: 12.75,
          14: 14.0, 16: 16.0, 18: 18.0, 20: 20.0, 24: 24.0}
MIN_LEN_M = 0.3            # короче — фитинги, арматура, текст
MIN_W_M, MAX_W_M = 0.06, 0.5
GAP_INK = 0.15             # доля тёмного внутри трубы (между линиями) — почти пусто


def nps_from_width(width_m: float) -> float:
    od_in = width_m / 0.0254
    return min(NPS_OD, key=lambda n: abs(NPS_OD[n] - od_in) / NPS_OD[n])


def _lines(dark: np.ndarray, axis: int, L: int) -> list[tuple[float, int, int]]:
    """Прямые линии вдоль оси: (поперечная координата, начало, конец). axis 0 — горизонтальные."""
    k = np.ones((1, L), np.uint8) if axis == 0 else np.ones((L, 1), np.uint8)
    m = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i][:4])
        if axis == 0 and h <= 3:
            out.append((y + h / 2, x, x + w))
        elif axis == 1 and w <= 3:
            out.append((x + w / 2, y, y + h))
    return sorted(out)


def find_pipes(gray: np.ndarray, px_per_m: float, region=None) -> list[dict]:
    """Участки труб из двойных линий. Возвращает [{"axis","c","a0","a1","width_px","length_m","nps","bbox"}]."""
    dark = (gray < 200).astype(np.uint8)
    if region:
        x0, y0, x1, y1 = (int(v) for v in region)
        mask = np.zeros_like(dark)
        mask[max(0, y0):y1, max(0, x0):x1] = 1
        dark &= mask
    L = max(8, int(MIN_LEN_M * px_per_m))
    wmin, wmax = MIN_W_M * px_per_m, MAX_W_M * px_per_m
    out = []
    for axis in (0, 1):
        lines = _lines(dark, axis, L)
        used = set()
        for i, s in enumerate(lines):
            if i in used:
                continue
            best = None
            for j in range(i + 1, len(lines)):
                t = lines[j]
                d = t[0] - s[0]
                if d > wmax:
                    break
                if d < wmin or j in used:
                    continue
                a0, a1 = max(s[1], t[1]), min(s[2], t[2])
                if a1 - a0 < L or a1 - a0 < 0.6 * min(s[2] - s[1], t[2] - t[1]):
                    continue
                # внутри трубы пусто: средняя линия между двумя почти белая
                mid = int(round((s[0] + t[0]) / 2))
                strip = dark[mid, a0:a1] if axis == 0 else dark[a0:a1, mid]
                if strip.mean() > GAP_INK:
                    continue
                # между линиями нет третьей длинной линии (иначе это стена, рама, лоток)
                if any(s[0] + 1 < u[0] < t[0] - 1 and min(a1, u[2]) - max(a0, u[1]) > 0.5 * (a1 - a0)
                       for u in lines[i + 1:j]):
                    continue
                best = (j, t, a0, a1, d)
                break                          # ближайшая подходящая пара
            if best is None:
                continue
            j, t, a0, a1, d = best
            used.update((i, j))
            c = (s[0] + t[0]) / 2
            w = d
            bb = [a0, c - w / 2, a1, c + w / 2] if axis == 0 else [c - w / 2, a0, c + w / 2, a1]
            out.append({"axis": "h" if axis == 0 else "v", "c": round(c, 1), "a0": int(a0), "a1": int(a1),
                        "width_px": round(w, 1), "length_m": round((a1 - a0) / px_per_m, 2),
                        "nps": nps_from_width(w / px_per_m), "bbox": [round(v, 1) for v in bb]})
    return out
