"""Run CubiCasa5k model once per PNG and cache compact results to disk.

python -m stoa.cubicasa_infer [--limit N] [--no-tta]

Per image saves cache/cubicasa/<name>.npz with:
  room_cls, room_prob   (H, W) uint8 argmax class / prob*255
  icon_cls, icon_prob   (H, W) uint8
  polygons.json: opening/icon polygons from the original post-processing (get_polygons)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config

sys.path.insert(0, str(config.CUBICASA_REPO))

ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry",
                "Railing", "Storage", "Garage", "Undefined"]
ICON_CLASSES = ["No Icon", "Window", "Door", "Closet", "Electrical Applience", "Toilet", "Sink", "Sauna Bench",
                "Fire Place", "Bathtub", "Chimney"]
N_CLASSES, SPLIT = 44, [21, 12, 11]

_model = None


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model():
    global _model
    if _model is None:
        from floortrans.models.hg_furukawa_original import hg_furukawa_original
        m = hg_furukawa_original(n_classes=51)  # skip init_weights(): it loads a relative-path pretrain file
        m.conv4_ = torch.nn.Conv2d(256, N_CLASSES, bias=True, kernel_size=1)
        m.upsample = torch.nn.ConvTranspose2d(N_CLASSES, N_CLASSES, kernel_size=4, stride=4)
        ck = torch.load(config.CUBICASA_WEIGHTS, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model_state"])
        _model = m.eval().to(device())
    return _model


def predict(img_path, tta: bool = True, max_side: int = 0, scale: float = 1.0) -> torch.Tensor:
    """Returns averaged raw prediction (1, 44, H, W) on CPU at (original size * scale).
    img_path may be a path or a PIL image."""
    from floortrans.loaders import RotateNTurns
    rot = RotateNTurns()
    model = load_model()
    pil = img_path.convert("RGB") if isinstance(img_path, Image.Image) else Image.open(img_path).convert("RGB")
    if scale != 1.0:
        pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.LANCZOS)
    im = np.asarray(pil, dtype=np.float32)
    H, W = im.shape[:2]
    x = torch.from_numpy(np.moveaxis(2 * (im / 255.0) - 1, -1, 0))[None].to(device())
    if max_side and max(H, W) > max_side:
        s = max_side / max(H, W)
        x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=True)
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)] if tta else [(0, 0)]
    acc = torch.zeros((1, N_CLASSES, H, W))
    with torch.no_grad():
        for fwd, back in rotations:
            p = model(rot(x, "tensor", fwd))
            p = rot(p, "tensor", back)
            p = rot(p, "points", back)
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=True)
            acc += p.cpu()
    return acc / len(rotations)


def to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    return o


def process(img_path: Path, out_dir: Path, tta=True, polygons=True, scale=1.0):
    from floortrans.post_prosessing import split_prediction, get_polygons
    pred = predict(img_path, tta, scale=scale)
    H, W = pred.shape[2:]
    rooms = F.softmax(pred[0, 21:33], 0)
    icons = F.softmax(pred[0, 33:], 0)
    rp, rc = rooms.max(0)
    ip, ic = icons.max(0)
    np.savez_compressed(out_dir / f"{img_path.stem}.npz",
                        room_cls=rc.numpy().astype(np.uint8), room_prob=(rp.numpy() * 255).astype(np.uint8),
                        icon_cls=ic.numpy().astype(np.uint8), icon_prob=(ip.numpy() * 255).astype(np.uint8),
                        scale=np.float32(scale))
    if polygons:
        try:
            heat, rms, ics = split_prediction(pred, (H, W), SPLIT)
            pols, types, room_pols, room_types = get_polygons((heat, rms, ics), 0.2, [1, 2])
            rec = {"polygons": to_jsonable(pols), "types": to_jsonable(types),
                   "room_polygons": [to_jsonable(np.array(p.exterior.coords)) if hasattr(p, "exterior") else None
                                     for p in room_pols],
                   "room_types": to_jsonable(room_types)}
        except Exception as e:
            rec = {"error": repr(e)}
        (out_dir / f"{img_path.stem}.polygons.json").write_text(json.dumps(rec))


# --- big sheets: crop to the drawing + scale to a physical px/m (see research_brief_large_plans.md, Г1/Г3) ---
BIG_SIDE = 4000        # original long side above which an image is treated as a large drawing sheet
PASS1_SIDE = 3500      # first pass resolution (crop detection + px/m estimate)
TARGET_PPM = 55        # px per metre for the final pass; toilets/sinks appear from ~42, stable at 50-60 (TEST-4)
MAX_PIXELS_BIG = 20e6  # memory/time cap for the final pass


def drawing_bbox(gray: np.ndarray, margin: float = 0.02):
    """Bounding box of the drawing on a sheet. Thick dark strokes (walls) survive a morphological opening; thin
    lines/text of the title block, notes and frame do not. Wall fragments of one building are merged by a dilation
    (~2% of the sheet), then building blocks at least 10% the size of the largest are kept (wings), small isolated
    thick blobs (logo, north arrow) are dropped."""
    import cv2
    dark = (gray < 100).astype(np.uint8)
    k = max(3, round(min(gray.shape) * 0.0015))
    op = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    d = max(5, round(max(gray.shape) * 0.02))
    blob = cv2.dilate(op, np.ones((d, d), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    if n <= 1:
        return 0, 0, gray.shape[1], gray.shape[0]
    big = st[1:, 4].max()
    keep = [i for i in range(1, n) if st[i, 4] >= 0.10 * big]
    # undo the dilation margin (d/2) on the blob boxes
    x0 = min(st[i, 0] for i in keep) + d // 2; y0 = min(st[i, 1] for i in keep) + d // 2
    x1 = max(st[i, 0] + st[i, 2] for i in keep) - d // 2; y1 = max(st[i, 1] + st[i, 3] for i in keep) - d // 2
    m = int(margin * max(gray.shape))
    return max(0, x0 - m), max(0, y0 - m), min(gray.shape[1], x1 + m), min(gray.shape[0], y1 + m)


def _save(out_path: Path, pred, **meta):
    rooms = F.softmax(pred[0, 21:33], 0)
    icons = F.softmax(pred[0, 33:], 0)
    rp, rc = rooms.max(0)
    ip, ic = icons.max(0)
    np.savez_compressed(out_path, room_cls=rc.numpy().astype(np.uint8), room_prob=(rp.numpy() * 255).astype(np.uint8),
                        icon_cls=ic.numpy().astype(np.uint8), icon_prob=(ip.numpy() * 255).astype(np.uint8), **meta)
    return ic.numpy().astype(np.uint8)


def process_auto(img_path: Path, out_dir: Path, tta=True) -> dict:
    """Small images: plain pass at scale 1. Large sheets: crop drawing, estimate px/m from doors, rescale."""
    from .cubicasa_count import estimate_px_per_m
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    if max(W, H) <= BIG_SIDE:
        process(img_path, out_dir, tta=tta, polygons=False)
        return {"big_sheet": False}
    s1 = PASS1_SIDE / max(W, H)
    small = pil.resize((round(W * s1), round(H * s1)), Image.LANCZOS)
    bx0, by0, bx1, by1 = drawing_bbox(np.asarray(small.convert("L")))
    ox, oy = int(bx0 / s1), int(by0 / s1)
    crop = pil.crop((ox, oy, min(W, int(bx1 / s1)), min(H, int(by1 / s1))))
    ic1 = _save(out_dir / f"{img_path.stem}.pass1.npz", predict(crop, tta=False, scale=s1), scale=np.float32(s1))
    ppm1 = estimate_px_per_m(ic1)
    ppm0 = ppm1 / s1 if ppm1 else None
    s = TARGET_PPM / ppm0 if ppm0 else 5000 / max(W, H)
    s = min(s, (MAX_PIXELS_BIG / (crop.width * crop.height)) ** 0.5)
    pred = predict(crop, tta=tta, scale=s)
    icf = F.softmax(pred[0, 33:], 0).argmax(0).numpy().astype(np.uint8)
    # px/m in final mask pixels: from pass-1 doors; else from final-pass doors; else assume the target
    ppm_final = (ppm0 * s) if ppm0 else (estimate_px_per_m(icf) or TARGET_PPM)
    _save(out_dir / f"{img_path.stem}.npz", pred, scale=np.float32(s), offset=np.array([ox, oy], dtype=np.int32),
          px_per_m=np.float32(ppm_final), big_sheet=np.bool_(True))
    return {"big_sheet": True, "crop": [ox, oy, ox + crop.width, oy + crop.height], "px_per_m_original": ppm0,
            "scale": s, "final_size": [round(crop.width * s), round(crop.height * s)],
            "final_px_per_m": round(float(ppm_final), 1), "capped": bool(ppm0 and ppm0 * s < 0.9 * TARGET_PPM)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-polygons", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="resize input by this factor before the model")
    ap.add_argument("--max-pixels", type=float, default=6e6, help="cap scaled image area")
    ap.add_argument("--list", help="text file with dataset-relative paths, one per line")
    ap.add_argument("--auto", action="store_true", help="large sheets: crop drawing + physical px/m scaling")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    out = config.CACHE_DIR / ("cubicasa" if a.scale == 1.0 else f"cubicasa_s{a.scale:g}")
    out.mkdir(parents=True, exist_ok=True)
    if a.list:
        a.files = [str(config.DATASET_DIR / l.strip()) for l in open(a.list) if l.strip()]
    files = [Path(f) for f in a.files] or sorted((config.DATASET_DIR / "data" / "cubicasa5k").glob("*.png"))
    if a.limit:
        files = files[: a.limit]
    t0 = time.time()
    for i, f in enumerate(files):
        if not a.force and (out / f"{f.stem}.npz").exists():
            continue
        t = time.time()
        if a.auto:
            info = process_auto(f, out, tta=not a.no_tta)
            print(f"[{i+1}/{len(files)}] {f.name} {info} {time.time()-t:.1f}s", flush=True)
            continue
        w, h = Image.open(f).size
        sc = min(a.scale, (a.max_pixels / (w * h)) ** 0.5) if a.scale > 1 else a.scale
        process(f, out, tta=not a.no_tta, polygons=not a.no_polygons, scale=sc)
        print(f"[{i+1}/{len(files)}] {f.name} {time.time()-t:.1f}s (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
