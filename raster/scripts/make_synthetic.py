"""Генератор синтетических планов в формате CubiCasa5K (F1_scaled.png + model.svg).

Нужен для тестов пайплайна и smoke-обучения, пока настоящий датасет не скачан.
    python scripts/make_synthetic.py --out data/synthetic --n 40
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

T = 10  # толщина стены


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def pts(p):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in p)


def make_plan(rnd: random.Random):
    W, H = rnd.randint(800, 1100), rnd.randint(600, 850)
    x0, y0, x1, y1 = 40, 40, W - 40, H - 40
    vx = rnd.randint(x0 + 250, x1 - 360)           # вертикальная перегородка
    hy = rnd.randint(y0 + 200, y1 - 200)           # горизонтальная перегородка слева
    bx = rnd.randint(vx + 160, x1 - 160)           # перегородка санузла справа
    by = rnd.randint(y0 + 160, hy)                 # низ санузла

    img = np.full((H, W, 3), 255, np.uint8)
    walls = [rect(x0, y0, x1, y0 + T), rect(x0, y1 - T, x1, y1), rect(x0, y0, x0 + T, y1), rect(x1 - T, y0, x1, y1),
             rect(vx, y0, vx + T, y1), rect(x0, hy, vx, hy + T), rect(vx, by, x1, by + T), rect(bx, y0, bx + T, by)]
    rooms = [("LivingRoom", rect(x0 + T, hy + T, vx, y1 - T)),
             ("Bedroom", rect(x0 + T, y0 + T, vx, hy)),
             ("Bath", rect(vx + T, y0 + T, bx, by)),
             ("Storage", rect(bx + T, y0 + T, x1 - T, by)),
             ("Kitchen", rect(vx + T, by + T, x1 - T, y1 - T))]
    for _, r in rooms:
        cv2.fillPoly(img, [np.int32(r)], (250, 250, 250))
    for w in walls:
        cv2.fillPoly(img, [np.int32(w)], (40, 40, 40))

    # двери: проём в стене + дуга
    doors = []
    for (wx, wy, horiz) in [(x0 + 60, hy, True), (vx, hy + 80, False), (vx, by - 90, False), (vx + 40, by, True)]:
        d = rect(wx, wy, wx + 70, wy + T) if horiz else rect(wx, wy, wx + T, wy + 70)
        cv2.fillPoly(img, [np.int32(d)], (255, 255, 255))
        if horiz:
            cv2.ellipse(img, (wx, wy + T), (70, 70), 0, 0, 90, (60, 60, 60), 1)
        else:
            cv2.ellipse(img, (wx + T, wy), (70, 70), 0, 0, 90, (60, 60, 60), 1)
        doors.append(d)

    # окна на внешних стенах
    windows = []
    for _ in range(rnd.randint(3, 6)):
        side = rnd.choice("tblr")
        if side in "tb":
            wx = rnd.randint(x0 + 30, x1 - 130)
            wy = y0 if side == "t" else y1 - T
            r = rect(wx, wy, wx + 90, wy + T)
        else:
            wy = rnd.randint(y0 + 30, y1 - 130)
            wx = x0 if side == "l" else x1 - T
            r = rect(wx, wy, wx + T, wy + 90)
        if any(cv2.pointPolygonTest(np.float32(w), (float(r[0][0] + 2), float(r[0][1] + 2)), False) >= 0 for w in windows):
            continue
        cv2.fillPoly(img, [np.int32(r)], (255, 255, 255))
        cv2.polylines(img, [np.int32(r)], True, (40, 40, 40), 1)
        cv2.line(img, (int((r[0][0] + r[2][0]) / 2) if side in "lr" else r[0][0], int((r[0][1] + r[2][1]) / 2) if side in "tb" else r[0][1]),
                 (int((r[0][0] + r[2][0]) / 2) if side in "lr" else r[2][0], int((r[0][1] + r[2][1]) / 2) if side in "tb" else r[2][1]), (40, 40, 40), 1)
        windows.append(r)

    # сантехника в санузле
    fixtures = []
    tx, ty = vx + T + 15, y0 + T + 10
    t = rect(tx, ty, tx + 40, ty + 60)
    cv2.rectangle(img, (tx + 5, ty), (tx + 35, ty + 15), (40, 40, 40), 1)
    cv2.ellipse(img, (tx + 20, ty + 38), (17, 22), 0, 0, 360, (40, 40, 40), 1)
    fixtures.append(("Toilet", t))
    sx, sy = tx + 60, ty
    s = rect(sx, sy, sx + 50, sy + 40)
    cv2.rectangle(img, (sx, sy), (sx + 50, sy + 40), (40, 40, 40), 1)
    cv2.ellipse(img, (sx + 25, sy + 22), (16, 12), 0, 0, 360, (40, 40, 40), 1)
    fixtures.append(("Sink", s))
    if bx - vx > 200 and by - y0 > 150:
        bt = rect(bx - 75, y0 + T + 8, bx - 8, y0 + T + 150)
        cv2.rectangle(img, bt[0], bt[2], (40, 40, 40), 1)
        cv2.rectangle(img, (bt[0][0] + 6, bt[0][1] + 6), (bt[2][0] - 6, bt[2][1] - 6), (40, 40, 40), 1)
        fixtures.append(("Bathtub", bt))
    else:
        sh = rect(bx - 80, by - 80, bx - 5, by - 5)
        cv2.rectangle(img, sh[0], sh[2], (40, 40, 40), 1)
        cv2.line(img, sh[0], sh[2], (40, 40, 40), 1)
        cv2.line(img, sh[1], sh[3], (40, 40, 40), 1)
        fixtures.append(("ShowerCab", sh))
    # раковина на кухне
    kx, ky = x1 - T - 70, y1 - T - 50
    k = rect(kx, ky, kx + 60, ky + 40)
    cv2.rectangle(img, k[0], k[2], (40, 40, 40), 1)
    fixtures.append(("Sink", k))

    # подписи комнат (как на реальных планах)
    for name, r in rooms:
        cx, cy = (r[0][0] + r[2][0]) // 2, (r[0][1] + r[2][1]) // 2
        cv2.putText(img, name.upper()[:6], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><g class="Model"><g class="Floor">']
    for w in walls:
        svg.append(f'<g id="Wall" class="Wall "><polygon points="{pts(w)}"/></g>')
    for wn in windows:
        svg.append(f'<g id="Window" class="Window Regular"><polygon points="{pts(wn)}"/></g>')
    for d in doors:
        svg.append(f'<g id="Door" class="Door Swing Beside"><polygon points="{pts(d)}"/></g>')
    for name, r in rooms:
        svg.append(f'<g class="Space {name}"><polygon points="{pts(r)}"/></g>')
    for name, f in fixtures:
        svg.append(f'<g class="FixedFurniture {name}"><g class="BoundaryPolygon"><polygon points="{pts(f)}"/></g></g>')
    svg.append('<g class="FixedFurniture Closet"><g class="BoundaryPolygon"><polygon points="1,1 2,1 2,2"/></g></g>')
    svg.append("</g></g></svg>")
    return img, "\n".join(svg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    for i in range(a.n):
        d = a.out / "synthetic" / f"{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        img, svg = make_plan(rnd)
        cv2.imwrite(str(d / "F1_scaled.png"), img)
        (d / "model.svg").write_text(svg)
    print(f"{a.n} планов в {a.out}")


if __name__ == "__main__":
    main()
