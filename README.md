# Floor plan MVP: скан → объекты → чат

```
скан → preprocess → YOLO-seg → маски → полигоны → scene.json / scene.svg
                                                     ↓
              чат → шаблонный разбор ($0) или Claude tool calling → highlight/count/... → подсветка SVG
```

## Структура

| Файл | Что делает |
|---|---|
| `fp/classes.py` | список классов MVP, маппинг имён CubiCasa → наши классы, синонимы для чата |
| `fp/cubicasa_svg.py` | парсер `model.svg` из CubiCasa5K |
| `fp/convert_yolo.py` | CubiCasa5K → датасет YOLO-seg + `floorplans.yaml` |
| `fp/visualize.py` | визуальная проверка разметки (полигоны поверх картинки) |
| `fp/preprocess.py` | deskew, обрезка полей, CLAHE, ограничение размера |
| `fp/infer.py` | картинка → граф сцены (моделью YOLO или из разметки `--gt-svg`) |
| `fp/scene.py` | граф сцены, привязка объектов к комнатам, JSON/SVG |
| `fp/query.py` | инструменты `highlight/count/list_rooms/objects_in/area/clear` + confidence по `confidence.md` |
| `fp/chat.py` | чат: шаблоны (`parsed_by=cache`) → Claude (`parsed_by=llm`) |
| `server/app.py`, `web/index.html` | FastAPI + страница: план, легенда, чат |
| `scripts/make_synthetic.py` | синтетические планы в формате CubiCasa (для тестов) |
| `scripts/inspect_cubicasa.py` | какие классы реально есть в датасете |

## Быстрый старт

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

### 1. Датасет
CubiCasa5K (~5 ГБ, Zenodo: https://zenodo.org/records/2613548) распаковать в `data/cubicasa5k/`.

```bash
.venv/bin/python scripts/inspect_cubicasa.py data/cubicasa5k --limit 500     # проверить маппинг классов
.venv/bin/python -m fp.convert_yolo --src data/cubicasa5k --out data/yolo
.venv/bin/python -m fp.visualize --data data/yolo --split train --n 20 --out out/check
```

### 2. Обучение (Mac M-серии: `device=mps`; на Colab/CUDA: `device=0`)
```bash
.venv/bin/yolo segment train model=yolo11s-seg.pt data=data/yolo/floorplans.yaml epochs=100 imgsz=1024 device=mps project=runs name=cubicasa
.venv/bin/yolo segment val model=<путь>/best.pt data=data/yolo/floorplans.yaml   # mAP по классам
```
Путь к `best.pt` печатается в конце обучения (Ultralytics 8.4 кладёт в `runs/segment/runs/<name>/weights/`).
Ниже `runs/cubicasa/weights/best.pt` — подставьте свой.

**Версии модели.** v1 — 15 классов (сауна внутри `room_bath`), обучена 11 эпох в Colab
(`MyDrive/floorplan/runs/cubicasa`). v2 — 16 классов: добавлен значок `sauna_bench` (все SaunaBench*: ~10,9 тыс. скамей в 3235 планах;
помещений с меткой Sauna всего 133 — сауну рисуют скамьями в Bath). v2 дообучается поверх v1 (`MyDrive/floorplan/runs/cubicasa_v2`): Ultralytics переносит голову
классов по именам. Новые классы добавлять **только в конец** `CLASSES`, затем
`model=<v1>/best.pt data=<разметка v2>` — новый запуск, не `resume`.

### 3. Инференс и чат
```bash
.venv/bin/python -m fp.infer --image my_scan.png --model runs/cubicasa/weights/best.pt --out data/scenes/my_scan
FP_MODEL=runs/cubicasa/weights/best.pt .venv/bin/uvicorn server.app:app --port 8000
```
Без модели можно посмотреть чат на эталонной разметке:
`python -m fp.infer --image <F1_scaled.png> --gt-svg <model.svg> --out data/scenes/demo`.

Для запросов, которые шаблоны не понимают, нужен `ANTHROPIC_API_KEY` (модель `claude-opus-5-5`,
effort `low`, включены server-side fallbacks на случай отказа модели).

## Confidence
Считается кодом, LLM не спрашиваем (см. `fp/query.py`):
- у элемента `conf` = уверенность YOLO (у GT-разметки 1.0);
- ответ = `min` по элементам; ноль найденных → ≤ 0.4; второй источник (`primary=False`, например OCR)
  совпал → ≥ 0.95, разошёлся → ≤ 0.4;
- уровни: `high ≥ 0.9`, `medium ≥ 0.7`, иначе `low` (+ примечание «проверьте»).

## Известные ограничения MVP
- Дверь между двумя комнатами привязывается к одной (по центру).
- Площадь в px², пока не задан масштаб (`meta.px_per_m` в scene.json).
- У YOLO маски бинарные, поэтому `conf` = уверенность бокса, а не средняя вероятность по маске.
- Второй источник (OCR) и VLM-арбитр для низкой уверенности — ещё не подключены.

### Пороги и склейка дублей (под точный подсчёт)
`fp/thresholds.json` — порог уверенности по классам и склейка перекрывающихся детекций одного класса
(одно окно, разбитое моделью на две). Подбираются на val, проверяются на test:
```bash
.venv/bin/python scripts/tune_thresholds.py dump --yolo <best.pt> --split val
.venv/bin/python scripts/tune_thresholds.py dump --yolo <best.pt> --split test
.venv/bin/python scripts/tune_thresholds.py tune      # -> out/tune/report.md, out/tune/thresholds.json
```
Новая модель (v2) → перезапустить и скопировать результат в `fp/thresholds.json`.
Сравнение с готовой моделью CubiCasa: `scripts/compare_models.py` (результаты в `out/compare/`).

### Калибровка уверенности
Уровень high/medium/low у подсчёта считается не по min(conf), а по числу **пограничных детекций**
(conf в пределах ±0.15 от порога класса, включая отброшенные) и min(conf). Границы по классам подбираются
на val так, чтобы ответы high были верны в ≥90% случаев, medium — в ≥70%:
`.venv/bin/python scripts/tune_thresholds.py calibrate` → `fp/thresholds.json`, отчёт `out/tune/calibration.md`.
На test (v2): high — 36% ответов, верно 89%; medium — 26%, верно 72%; low — 37%, верно 42%.

### ИИ в чате (Claude)
Ключ — в `.env` (образец `.env.example`, файл в `.gitignore`): `ANTHROPIC_API_KEY=...`.
Сначала вопрос разбирают правила (`parsed_by: cache`, $0); не поняли — Claude **Haiku 5.5** выбирает функцию
(count / highlight / objects_in / area / list_rooms), числа и id берутся из базы (`answered_by: db`).
Отвечает на языке вопроса; ~2–3 с и ~$0.0004 за вопрос. Другая модель: `FP_CLAUDE_MODEL=claude-opus-5-5`.

### Площади помещений и плинтусы
При загрузке (после чтения листа, `fp/sheet.py`) для каждой подписанной комнаты (`fp/spaces.py`):
площадь = подпись на плане, если есть; иначе геометрия — заливка по стенам от подписи (`fp/rooms_cv.py`, без
нейросети) или контур комнаты модели. Два источника совпали в пределах 10% — high, один — medium, разошлись — low.
Плинтус = периметр (для почти прямоугольных комнат — периметр описанного прямоугольника) минус ширина дверей,
касающихся комнаты. Масштаб: шкала, подписи площадей, или текст масштаба + формат листа (1:200 на A2).
Чат: «посчитай площади помещений и плинтусы», «плинтус в MH», англ. «room areas / baseboards».
