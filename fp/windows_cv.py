"""Окна по полосам остекления в стенах — без нейросети, для чертёжных листов.

На архитектурном листе окно — участок стены, где вместо сплошной заливки идут 2–3 тонкие линии остекления
вдоль стены, между простенками. Модель YOLO обучена на квартирах CubiCasa, где окна рисуют иначе, и на таких
листах находит их частично (школа C01: 27 из ~60).
1. Стены — толстые тёмные штрихи (fp.rooms_cv.wall_mask): тонкие линии остекления туда не попадают.
2. Тонкие линии длиной ≥ 0.4 м вне стен, сгруппированные по 2–3 параллельно с шагом не больше толщины стены.
3. Оба конца полосы упираются в стену или простенок — иначе это мебель; групп больше трёх линий — лестницы.
На квартирах CubiCasa этот способ вредит (лишние окна), поэтому включается только на чертёжных листах.
"""
from __future__ import annotations

import cv2
import numpy as np

from .rooms_cv import wall_mask

WIN_MIN_M, WIN_MAX_M = 0.4, 4.0
THIN = 235                # тонкие линии остекления бывают светло-серыми (~210-225)


def find_glazing(gray: np.ndarray, px_per_m: float) -> list[dict]:
    """Окна как полосы остекления: 2–3 тонкие параллельные линии длиной ≥ WIN_MIN_M на расстоянии не больше
    толщины стены друг от друга, оба конца полосы упираются в стену или простенок. Работает и для фасадов,
    где окна стоят между квадратными простенками (у них нет торца, от которого строится луч)."""
    walls, thick = wall_mask(gray)
    wall_d = cv2.dilate(walls, np.ones((3, 3), np.uint8))
    thin = ((gray < THIN) & (wall_d == 0)).astype(np.uint8)
    L = max(5, int(WIN_MIN_M * px_per_m))
    Lmax = WIN_MAX_M * px_per_m
    reach = max(3, int(0.6 * thick))          # насколько за концом полосы ищем стену
    h, w = gray.shape
    out = []
    for axis in (0, 1):                       # 0 — вертикальные полосы, 1 — горизонтальные
        k = np.ones((L, 1), np.uint8) if axis == 0 else np.ones((1, L), np.uint8)
        segs_mask = cv2.morphologyEx(thin, cv2.MORPH_OPEN, k)
        n, lab, st, _ = cv2.connectedComponentsWithStats(segs_mask, connectivity=8)
        segs = []                              # (поперечная координата, начало, конец)
        for i in range(1, n):
            x, y, bw, bh, _ = st[i]
            if axis == 0 and bw <= 3:
                segs.append((x + bw / 2, y, y + bh))
            elif axis == 1 and bh <= 3:
                segs.append((y + bh / 2, x, x + bw))
        # группы параллельных линий: перекрытие по длине ≥ 70%, шаг ≤ толщины стены
        segs.sort()
        used = [False] * len(segs)
        for i, s in enumerate(segs):
            if used[i]:
                continue
            group = [s]
            used[i] = True
            for j in range(i + 1, len(segs)):
                t = segs[j]
                if t[0] - group[-1][0] > 1.3 * thick:
                    break
                ov = min(s[2], t[2]) - max(s[1], t[1])
                if not used[j] and ov >= 0.7 * min(s[2] - s[1], t[2] - t[1]) and t[0] - group[-1][0] >= 2:
                    group.append(t)
                    used[j] = True
            if not 2 <= len(group) <= 3:
                continue                       # одна линия — не остекление; много — лестница, штриховка
            c = sum(g[0] for g in group) / len(group)
            a0 = int(np.median([g[1] for g in group]))
            a1 = int(np.median([g[2] for g in group]))
            if a1 - a0 > Lmax:
                continue
            ci = int(round(c))

            def wall_at(p0, p1):
                p0, p1 = max(0, p0), min((h if axis == 0 else w) - 1, p1)
                if p1 <= p0:
                    return False
                lo, hi = max(0, ci - 2), ci + 3
                band = walls[p0:p1, lo:hi] if axis == 0 else walls[lo:hi, p0:p1]
                return band.any()
            if not (wall_at(a0 - reach, a0 + 2) and wall_at(a1 - 2, a1 + reach)):
                continue
            half = (group[-1][0] - group[0][0]) / 2 + 2
            if axis == 0:
                bb = [c - half, a0, c + half, a1]
            else:
                bb = [a0, c - half, a1, c + half]
            out.append({"bbox": [round(v, 1) for v in bb], "length_m": round((a1 - a0) / px_per_m, 2),
                        "lines": len(group)})
    return out
