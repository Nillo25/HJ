# Make Blueprints Talk — North Line

Вопросы на естественном языке о строительных чертежах и BIM-моделях.
Точно, дёшево и с показом, откуда взят ответ.

Junction 2026, трек Stoa.

## Идея

Считать, а не предполагать. Где ответ лежит в структуре файла — его
вычисляет код: геометрия DXF, элементы IFC, текстовый слой PDF. Модель
читает вопрос и формулирует ответ, но числа не производит. Зрение
включается только там, где больше не на что смотреть — на растровых планах.

Стоимость растёт с числом файлов, а не вопросов: файл разбирается один раз,
вопрос становится запросом к готовой базе.

## Устройство

```
indexer/   Python: разбор файлов в SQLite. Запускается отдельно.
server/    Hono: API поверх базы, разбор вопроса, отдача фронта.
web/       React + shadcn: просмотрщик с подсветкой, панель точности.
docs/      Контракт API и схема базы.
```

## Запуск

```bash
# 1. индексация
cd indexer
pip install -r requirements.txt
python index.py ~/Downloads/junction-dataset-preview --db ../app.db

# 2. точность
python eval.py ~/Downloads/junction-dataset-preview --db ../app.db

# 3. сервер
cd ../server && npm i && npm run dev
```

## Run

    pip install -r indexer/requirements.txt        # pdfminer.six only
    cd server && npm i && cp .env.example .env     # add a model key, optional
    cd ../web && npm i && npm run build            # builds into server/public
    cd ../server && npm start                      # http://localhost:3000

Upload a drawing set in the UI (DXF + PNG/PDF with the same name, IFC, or a PDF).
Known question patterns are answered without a model; anything else goes to the
model set in `.env`. Token use per model is shown in the page header.
