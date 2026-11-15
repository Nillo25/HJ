"""FastAPI: загрузка скана -> граф сцены, отдача SVG/JSON, чат.

    FP_MODEL=runs/segment/train/weights/best.pt .venv/bin/uvicorn server.app:app --reload
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from fp import cache as answer_cache
from fp import infer
from fp.chat import chat
from fp.classes import COLORS, LABELS_RU
from fp.scene import Scene

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path):
    """Минимальный .env: KEY=VALUE построчно; переменные окружения имеют приоритет."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:  # пустые строки из образца не должны перекрывать заполненные
                    os.environ.setdefault(k.strip(), v)


_load_env(ROOT / ".env")
SCENES = Path(os.environ.get("FP_SCENES", ROOT / "data" / "scenes"))
# v2: 16 классов, 5-я эпоха; пороги в fp/thresholds.json; передача весов — WEIGHTS.md
_DEFAULT_MODELS = [ROOT / "runs/colab/floorplan_v2.pt", ROOT / "runs/colab/v2_best.pt"]
MODEL = os.environ.get("FP_MODEL") or next((str(p) for p in _DEFAULT_MODELS if p.exists()), None)
SCENES.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Floor plan MVP")
answer_cache.configure(SCENES / "cache.sqlite")


def _pipeline() -> str:
    """Отпечаток разбора: веса (имя, размер, время) + пороги + версия кода. Сменился — файл разбирается заново."""
    parts = [infer.PIPELINE_VERSION, (ROOT / "fp" / "thresholds.json").read_text()]
    if MODEL and Path(MODEL).exists():
        st = Path(MODEL).stat()
        parts.append(f"{Path(MODEL).name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _dir(plan_id: str) -> Path:
    if not re.fullmatch(r"[\w\-]+", plan_id):
        raise HTTPException(400, "bad id")
    d = SCENES / plan_id
    if not (d / "scene.json").exists():
        raise HTTPException(404, "план не найден")
    return d


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text()


@app.get("/api/meta")
def meta():
    return {"model": MODEL, "llm": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
            "colors": COLORS, "labels": LABELS_RU}


@app.get("/api/cache")
def cache_stats():
    return answer_cache.stats()


@app.get("/api/plans")
def plans():
    return sorted(p.name for p in SCENES.iterdir() if (p / "scene.json").exists())


@app.post("/api/cache/clear")
def clear_cache():
    """Сброс кэша: разобранные планы и кэш ответов чата удаляются (резервных копий не храним — они копились
    и раздували папку). Следующая загрузка каждого файла разберёт его заново. Словарь пользователя не трогаем."""
    import shutil
    items = list(SCENES.iterdir())
    n_plans = sum(1 for p in items if (p / "scene.json").exists())
    for p in items:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return {"removed": len(items), "plans": n_plans}


@app.post("/api/plans")
async def upload(file: UploadFile = File(...), force: bool = False):
    if not MODEL:
        raise HTTPException(503, "модель не задана: запустите сервер с FP_MODEL=<путь к best.pt>")
    suffix = Path(file.filename or "x.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        raise HTTPException(400, "нужна картинка PNG/JPG")
    data = await file.read()
    sha = hashlib.sha256(data).hexdigest()
    pipe = _pipeline()
    # тот же файл уже разобран той же моделью и порогами — отдаём готовую сцену: повтор бесплатен
    if not force:
        for d in sorted(SCENES.glob(f"*-{sha[:10]}")):
            try:
                m = json.loads((d / "scene.json").read_text()).get("meta", {})
            except (OSError, ValueError):
                continue
            if m.get("file_sha256") == sha and m.get("pipeline") == pipe:
                if not (d / "image_hd.png").exists():  # сцена старше копии для показа — достроить, без разбора
                    await run_in_threadpool(_display_from_upload, data, d)
                return {"id": d.name, "cached": True}
    stem = re.sub(r"[^\w\-]+", "_", Path(file.filename or "plan").stem)[:40]
    # id = имя + начало sha256 файла: тот же файл -> тот же id, на любой машине
    plan_id = next((d.name for d in sorted(SCENES.glob(f"*-{sha[:10]}"))), f"{stem}-{sha[:10]}")
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            # разбор листа — 15-35 с (YOLO, вызовы ИИ, дуги): в отдельном потоке, иначе обработчик async
            # держит цикл событий и сервер не отвечает ни на какие запросы, пока идёт загрузка
            sc = await run_in_threadpool(infer.run, Path(tmp.name), SCENES / plan_id, model=MODEL)
        except ValueError as e:
            raise HTTPException(400, str(e))
    sc.meta.update(file_sha256=sha, pipeline=pipe, source_image=file.filename or sc.meta.get("source_image"))
    sc.save(SCENES / plan_id)
    if os.environ.get("FP_STOA") == "1":  # оригинал загрузки для stoa/: её цифры измерены на исходных картинках
        (SCENES / plan_id / f"src_{plan_id}{suffix}").write_bytes(data)
    return {"id": plan_id, "cached": False}


def _display_from_upload(data: bytes, d: Path) -> None:
    """Копия для показа из загруженного файла: та же предобработка (поворот, обрезка), что и при разборе."""
    import cv2
    import numpy as np
    from fp.preprocess import preprocess
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    keep: dict = {}
    _, info = preprocess(img, keep=keep)
    size = json.loads((d / "scene.json").read_text()).get("meta", {}).get("size")
    if keep.get("display") is not None and (not size or list(size) == info["size"]) \
            and max(keep["display"].shape[:2]) > max(info["size"]):  # та же обрезка, что у сцены, и крупнее
        infer.save_display(keep["display"], d)


@app.get("/api/plans/{plan_id}")
def scene(plan_id: str):
    d = Scene.load(_dir(plan_id)).to_dict()
    # разобрана другой версией разбора (старше трубопроводов, масштаба…) — интерфейс предложит разобрать заново
    d["stale"] = d.get("meta", {}).get("pipeline") != _pipeline()
    return d


@app.post("/api/plans/{plan_id}/reanalyze")
async def reanalyze(plan_id: str):
    """Разобрать сохранённую сцену заново текущей версией — по её картинке, без повторной загрузки файла."""
    if not MODEL:
        raise HTTPException(503, "модель не задана: запустите сервер с FP_MODEL=<путь к best.pt>")
    d = _dir(plan_id)
    old = json.loads((d / "scene.json").read_text()).get("meta", {})
    with tempfile.TemporaryDirectory() as tmp:  # infer.run перезаписывает image.png в папке сцены
        src = Path(tmp) / "image.png"
        hd = d / "image_hd.png"  # исходник крупнее 2048 px — копия для показа, иначе сама картинка разбора
        src.write_bytes((hd if hd.exists() else d / "image.png").read_bytes())
        try:
            sc = await run_in_threadpool(infer.run, src, d, model=MODEL)
        except ValueError as e:
            raise HTTPException(400, str(e))
    sc.meta.update(pipeline=_pipeline(), source_image=old.get("source_image") or sc.meta.get("source_image"))
    if old.get("file_sha256"):
        sc.meta["file_sha256"] = old["file_sha256"]
    sc.save(d)
    return {"id": plan_id, "cached": False}


# картинка и SVG меняются при повторном разборе под тем же адресом: браузер сверяет ETag, а не берёт старую копию
NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/api/plans/{plan_id}/image")
def image(plan_id: str):
    d = _dir(plan_id)
    hd = d / "image_hd.png"  # для показа — копия в высоком разрешении, координаты сцены те же (SVG её масштабирует)
    return FileResponse(hd if hd.exists() else d / "image.png", headers=NO_CACHE)


@app.get("/api/plans/{plan_id}/svg")
def svg(plan_id: str):
    return FileResponse(_dir(plan_id) / "scene.svg", media_type="image/svg+xml", headers=NO_CACHE)


class ChatIn(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/plans/{plan_id}/chat")
def chat_ep(plan_id: str, body: ChatIn):
    sc = Scene.load(_dir(plan_id))
    if os.environ.get("FP_STOA") == "1":  # вопросы-подсчёты Stoa по всему плану -> stoa/ (см. MERGE_STOA.md)
        from stoa import hook
        try:
            r = hook.answer(_dir(plan_id), body.message)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"stoa: {type(e).__name__}: {e}")
        if r:
            return r
    try:
        out = chat(sc, body.message, body.history)
    except Exception as e:  # noqa: BLE001 — показываем ошибку в чате, а не 500 без текста
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    # пользователь исправил план в чате (fp/corrections.py) — сохранить сцену; интерфейс перечитает её
    if sc.meta.pop("_dirty", None):
        sc.save(_dir(plan_id))
        out["scene_changed"] = True
    return out


class ScaleIn(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    unit: str = "m"


def _set_scale(plan_id: str, est: dict) -> dict:
    d = _dir(plan_id)
    sc = Scene.load(d)
    sc.meta["scale"] = est
    sc.meta["px_per_m"] = est.get("px_per_m")
    sc.save(d)
    return {k: v for k, v in est.items() if k != "vlm_read"}


@app.post("/api/plans/{plan_id}/scale")
def set_scale(plan_id: str, body: ScaleIn):
    """Ручной масштаб: две точки на плане (в пикселях картинки) и реальное расстояние между ними."""
    from fp.scale import UNIT_M, from_two_points
    if body.unit not in UNIT_M:
        raise HTTPException(400, f"единица {body.unit!r}; доступны: {', '.join(UNIT_M)}")
    try:
        est = from_two_points((body.x1, body.y1), (body.x2, body.y2), body.length, body.unit)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _set_scale(plan_id, est)


@app.post("/api/plans/{plan_id}/scale/auto")
def auto_scale(plan_id: str):
    """Перечитать лист уже загруженного плана: VLM один раз читает штамп, легенду, таблицы, шкалу, подписи;
    затем масштаб и зоны легенды. Возвращает масштаб."""
    import cv2
    from fp.infer import analyze_sheet
    d = _dir(plan_id)
    img = cv2.imread(str(d / "image.png"))
    # крупный лист: тайлы, проверка дверей и трубы — по копии высокого разрешения (та же обрезка), иначе
    # повторный анализ по 2048 px терял найденное при загрузке (M-101A 10200 px: труб 88 -> 9)
    hd = cv2.imread(str(d / "image_hd.png")) if (d / "image_hd.png").exists() else None
    f = img.shape[1] / hd.shape[1] if hd is not None else None
    sc = analyze_sheet(img, Scene.load(d), full=hd, f=f)
    sc.save(d)
    return sc.meta.get("scale", {})
