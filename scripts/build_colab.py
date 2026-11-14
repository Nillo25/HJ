"""Собирает colab/train_cubicasa.ipynb: код fp/ вшивается через %%writefile,
поэтому ноутбук самодостаточный. Перезапускать после правок в fp/.

    python scripts/build_colab.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["fp/__init__.py", "fp/classes.py", "fp/cubicasa_svg.py", "fp/convert_yolo.py", "fp/visualize.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# Floor plan MVP — обучение YOLO-seg на CubiCasa5K (GPU)

1. **Среда выполнения → Сменить среду выполнения → GPU** (T4 бесплатно; L4/A100 в Colab Pro).
2. Запускайте ячейки по порядку. Веса и логи пишутся на Google Drive в `MyDrive/floorplan/runs`,
   поэтому если сессия оборвётся — просто запустите всё заново: обучение **продолжится** с последней эпохи.
3. Итог: `MyDrive/floorplan/runs/cubicasa_v2/weights/best.pt` — скачать и положить в проект,
   затем `FP_MODEL=<путь>/best.pt uvicorn server.app:app`.
"""),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 1. Google Drive и зависимости"),
    code("""
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE = '/content/drive/MyDrive/floorplan'
os.makedirs(DRIVE, exist_ok=True)
!pip -q install ultralytics shapely lxml
"""),
    md("## 2. Код проекта (конвертер CubiCasa → YOLO)\nСгенерировано из `fp/` скриптом `scripts/build_colab.py`."),
    code("import os\nos.makedirs('/content/fp', exist_ok=True)\n%cd /content"),
]
for f in FILES:
    cells.append(code(f"%%writefile /content/{f}\n" + ((ROOT / f).read_text() or "# package\n")))

cells += [
    md("## 3. Датасет CubiCasa5K (5.5 ГБ, Zenodo)\nКачается на локальный диск Colab — это быстро. "
       "`KEEP_ZIP_ON_DRIVE = True` сохранит архив на Drive, чтобы в следующий раз не качать (нужно ~5.5 ГБ места на Drive)."),
    code("""
KEEP_ZIP_ON_DRIVE = False
URL = 'https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1'
ZIP_DRIVE = f'{DRIVE}/cubicasa5k.zip'
ZIP = ZIP_DRIVE if os.path.exists(ZIP_DRIVE) else '/content/cubicasa5k.zip'

if not os.path.exists('/content/data/cubicasa5k'):
    if not os.path.exists(ZIP):
        !wget -q --show-progress -c -O "$ZIP" "$URL"
    !mkdir -p /content/data && unzip -q "$ZIP" -d /content/data
    if KEEP_ZIP_ON_DRIVE and ZIP != ZIP_DRIVE:
        !cp "$ZIP" "$ZIP_DRIVE"
!ls /content/data/cubicasa5k && du -sh /content/data/cubicasa5k
"""),
    md("## 4. Конвертация в YOLO-seg + визуальная проверка"),
    code("""
%cd /content
# yolo2 = разметка v2 (16 классов, добавлена room_sauna); старая /content/data/yolo — v1, 15 классов
if not os.path.exists('/content/data/yolo2/floorplans.yaml'):
    !python -m fp.convert_yolo --src /content/data/cubicasa5k --out /content/data/yolo2 2>&1 | grep -v libpng
!for s in train val test; do echo $s $(ls /content/data/yolo2/images/$s | wc -l); done
"""),
    code("""
!python -m fp.visualize --data /content/data/yolo2 --split val --n 4 --out /content/check
from IPython.display import Image, display
import glob
for p in sorted(glob.glob('/content/check/*.jpg'))[:4]:
    display(Image(p, width=700))
"""),
    md("""
## 5. Обучение (v2 = дообучение поверх v1)
- Есть чекпоинт v2 (`runs/cubicasa_v2/weights/last.pt`) — продолжаем его (сессия оборвалась — просто запустите всё заново).
- Иначе есть веса v1 (`runs/cubicasa/weights/best.pt`, 15 классов) — **дообучаем** их на разметке v2:
  Ultralytics переносит голову классов по именам, учится заново только новый класс `room_sauna`.
- Иначе — с нуля от `yolo11s-seg.pt`.
- `batch=-1` — пачка под ~60% памяти GPU; `patience=15` — ранняя остановка.
"""),
    code("""
from ultralytics import YOLO
RUN = f'{DRIVE}/runs/cubicasa_v2'
last = f'{RUN}/weights/last.pt'
v1 = f'{DRIVE}/runs/cubicasa/weights/best.pt'

if os.path.exists(last):
    print('Продолжаю v2 с', last)
    model = YOLO(last)
    model.train(resume=True)
else:
    start = v1 if os.path.exists(v1) else 'yolo11s-seg.pt'
    print('Старт v2 от', start)
    model = YOLO(start)
    model.train(data='/content/data/yolo2/floorplans.yaml', epochs=60, imgsz=1024, batch=-1,
                device=0, workers=8, patience=15, cache=False, max_det=1000,
                project=f'{DRIVE}/runs', name='cubicasa_v2', exist_ok=True, plots=True)
"""),
    md("## 6. Метрики по классам на test (mAP масок)"),
    code("""
best = YOLO(f'{RUN}/weights/best.pt')
m = best.val(data='/content/data/yolo2/floorplans.yaml', split='test', imgsz=1024, device=0, plots=True, max_det=1000,
             project=f'{DRIVE}/runs', name='cubicasa_v2_test', exist_ok=True)
import pandas as pd
rows = [{'class': best.names[c], 'mask_mAP50': m.seg.ap50[i], 'mask_mAP50-95': m.seg.ap[i],
         'box_mAP50': m.box.ap50[i]} for i, c in enumerate(m.ap_class_index)]
pd.DataFrame(rows).round(3)
"""),
    md("## 7. Пример предсказания и скачивание весов"),
    code("""
import random
val_imgs = sorted(glob.glob('/content/data/yolo2/images/test/*.png'))
res = best.predict(random.choice(val_imgs), imgsz=1024, conf=0.25, retina_masks=True, max_det=1000)[0]
res.save('/content/pred.jpg')
display(Image('/content/pred.jpg', width=900))
"""),
    code("""
from google.colab import files
files.download(f'{RUN}/weights/best.pt')
"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}}
out = ROOT / "colab" / "train_cubicasa.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(out)
