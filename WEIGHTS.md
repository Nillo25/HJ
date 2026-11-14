# Веса модели

Веса в git не хранятся (`*.pt` в `.gitignore`). Они выложены в GitHub Release:
**https://github.com/Nillo25/HJ/releases/tag/weights-v2**

```bash
mkdir -p runs/colab && gh release download weights-v2 -R Nillo25/HJ -p floorplan_v2.pt -D runs/colab
# или без gh:
curl -L -o runs/colab/floorplan_v2.pt https://github.com/Nillo25/HJ/releases/download/weights-v2/floorplan_v2.pt
```

## Файл

| | |
|---|---|
| Имя | `floorplan_v2.pt` |
| Куда положить | `runs/colab/floorplan_v2.pt` (путь от корня репозитория) |
| Размер | 20,6 МБ (20 559 484 байт) |
| SHA-256 | `7e637dadb1a886f0b14468dc3da74c230d5b7a59cd4d6f2b399df0953f6f77e5` |
| Архитектура | YOLO11s-seg (Ultralytics 8.4.212), вход `imgsz=1024` |
| Обучение | CubiCasa5K, разбиение train/val/test из датасета (4200/400/400); v1 — 11 эпох, затем v2 — дообучение поверх v1, 5-я эпоха (лучшая) |
| Качество (val) | маски mAP50 0,767, mAP50-95 0,507 |

Это облегчённая копия `best.pt` из Colab: без состояния оптимизатора, предсказания те же.
Продолжать обучение с неё нельзя — для этого нужен `last.pt` с Drive (`MyDrive/floorplan/runs/cubicasa_v2/weights/`).

## Классы (порядок = id, не менять)

```
0 wall  1 door  2 window  3 toilet  4 sink  5 bathtub  6 shower
7 room_living  8 room_bedroom  9 room_kitchen  10 room_bath  11 room_entry
12 room_storage  13 room_outdoor  14 room_other  15 sauna_bench
```

Совпадает с `CLASSES` в `fp/classes.py`. Новые классы добавлять только в конец.

## Установка у коллеги

```bash
git clone https://github.com/Nillo25/HJ.git && cd HJ
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p runs/colab && curl -L -o runs/colab/floorplan_v2.pt https://github.com/Nillo25/HJ/releases/download/weights-v2/floorplan_v2.pt
shasum -a 256 runs/colab/floorplan_v2.pt        # должен совпасть с SHA-256 выше
cp .env.example .env                            # вписать свой ANTHROPIC_API_KEY (необязательно)
.venv/bin/python -m pytest -q tests             # все тесты должны пройти
.venv/bin/python -m uvicorn server.app:app --port 8000
```

Сервер берёт модель так: переменная `FP_MODEL`, иначе `runs/colab/floorplan_v2.pt`, иначе `runs/colab/v2_best.pt`.
Другой файл весов: `FP_MODEL=/путь/к/best.pt .venv/bin/python -m uvicorn server.app:app --port 8000`.

Без `ANTHROPIC_API_KEY` работает распознавание и чат на правилах (русские шаблонные вопросы).
С ключом добавляются: чтение листа (масштаб, легенда, таблицы, подписи комнат), площади и плинтусы,
вопросы на любом языке. Это около $0.002 за загрузку листа и около $0.0005 за вопрос через ИИ.

## Пороги и уверенность привязаны к весам

`fp/thresholds.json` (пороги по классам, склейка дублей, калибровка уверенности) подобран **под этот файл**.
С другими весами пороги нужно подобрать заново (нужен датасет CubiCasa5K, сконвертированный в `data/yolo`):

```bash
.venv/bin/python scripts/tune_thresholds.py dump --yolo <веса> --split val
.venv/bin/python scripts/tune_thresholds.py dump --yolo <веса> --split test
.venv/bin/python scripts/tune_thresholds.py tune        # затем перенести out/tune/thresholds.json в fp/thresholds.json
.venv/bin/python scripts/tune_thresholds.py calibrate   # дописывает калибровку в fp/thresholds.json
```

## Проверка в коде

```python
from ultralytics import YOLO
m = YOLO("runs/colab/floorplan_v2.pt")
r = m.predict("plan.png", imgsz=1024, conf=0.1, max_det=1000, retina_masks=True)[0]
print({m.names[int(c)] for c in r.boxes.cls})
```

Для подсчёта используйте `fp.infer.detect_yolo`: в нём пороги по классам и склейка дублей.
