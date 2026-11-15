"""Препроцессинг скана: обрезка полей, выравнивание (deskew), контраст, нормализация размера."""
from __future__ import annotations

import cv2
import numpy as np


def _content_mask(gray: np.ndarray) -> np.ndarray:
    # тёмное на светлом: Otsu, инвертируем
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Угол перекоса по длинным линиям (Хаф). Планы почти всегда из горизонталей/вертикалей."""
    edges = cv2.Canny(gray, 50, 150)
    min_len = max(gray.shape) // 8
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=100, minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # форма (N,1,4) или (N,4) в разных версиях OpenCV
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        a = (a + 45) % 90 - 45  # приводим к отклонению от ближайшей оси
        if abs(a) <= max_angle:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


DISPLAY_SIDE = 8192  # копия для показа: подписи на крупных листах (школа 14000 px) читаются, разбор — на max_side


def preprocess(img: np.ndarray, max_side: int = 2048, margin: int = 16,
               keep: dict | None = None) -> tuple[np.ndarray, dict]:
    """Возвращает (BGR картинка, инфо о преобразованиях). keep — если передан, в keep["display"] кладётся
    та же картинка (поворот, обрезка, контраст) в высоком разрешении, до DISPLAY_SIDE по длинной стороне."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # прозрачный фон -> белый
        alpha = img[:, :, 3:] / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    info = {"orig_size": [int(img.shape[1]), int(img.shape[0])]}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew(gray)
    if abs(angle) > 0.2:
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    info["deskew_deg"] = round(angle, 2)

    ys, xs = np.nonzero(_content_mask(gray))
    if len(xs) > 100:
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, img.shape[1])
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, img.shape[0])
        img = img[y0:y1, x0:x1]
        info["crop"] = [int(x0), int(y0), int(x1), int(y1)]

    # мягкое выравнивание контраста (CLAHE по яркости) — помогает на бледных сканах
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if keep is not None:
        d = min(1.0, DISPLAY_SIDE / max(img.shape[:2]))
        keep["display"] = cv2.resize(img, None, fx=d, fy=d, interpolation=cv2.INTER_AREA) if d < 1 else img.copy()
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    info["scale"] = round(min(s, 1.0), 4)
    info["size"] = [int(img.shape[1]), int(img.shape[0])]
    return img, info
