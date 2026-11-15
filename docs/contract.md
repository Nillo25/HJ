# Stoa — схема базы и контракт API

Черновик на ревью. Правим сейчас, дальше **не меняем** — на это завязаны все трое.

Принцип: парсеры кладут в базу **всё, что нашли в файле**, независимо от вопросов.
Вопрос — это фильтр поверх сохранённого. Иначе нечем отвечать на то, чего нет
в `questions.jsonl`.

---

## 1. База (SQLite)

```sql
-- Чертёж = группа файлов с одним именем (правило самого датасета)
CREATE TABLE drawings (
  id            INTEGER PRIMARY KEY,
  name          TEXT NOT NULL,          -- 'site1000_bldg2082_floor9385'
  source        TEXT NOT NULL,          -- 'swiss_dwellings' | 'ifc_models' | ...
  status        TEXT NOT NULL,          -- 'pending' | 'indexing' | 'ready' | 'failed'
  capabilities  TEXT,                   -- JSON: на что можем отвечать, см. §3
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Физический файл
CREATE TABLE files (
  id          INTEGER PRIMARY KEY,
  drawing_id  INTEGER NOT NULL REFERENCES drawings(id),
  path        TEXT NOT NULL,
  sha256      TEXT NOT NULL UNIQUE,     -- тот же файл не парсим дважды
  format      TEXT NOT NULL,            -- 'dxf' | 'ifc' | 'pdf' | 'png'
  role        TEXT NOT NULL,            -- 'truth' (отвечаем отсюда) | 'display' | 'ignored'
  width       REAL,                     -- для png/pdf: пиксели / пункты
  height      REAL,
  transform   TEXT,                     -- JSON {sx, sy, dx, dy, flipY} источник → display
  status      TEXT NOT NULL             -- 'ok' | 'failed'
);

-- Этаж (IFC: IfcBuildingStorey; DXF: один файл = один этаж)
CREATE TABLE floors (
  id          INTEGER PRIMARY KEY,
  drawing_id  INTEGER NOT NULL REFERENCES drawings(id),
  name        TEXT,                     -- '3.korrus', 'Kælder', 'Grundriss 2. OG'
  level       REAL,                     -- отметка, метры
  ord         INTEGER
);

-- ГЛАВНАЯ ТАБЛИЦА. Всё найденное, одним форматом.
CREATE TABLE elements (
  id          INTEGER PRIMARY KEY,
  drawing_id  INTEGER NOT NULL REFERENCES drawings(id),
  file_id     INTEGER NOT NULL REFERENCES files(id),
  floor_id    INTEGER REFERENCES floors(id),
  type        TEXT NOT NULL,            -- см. словарь §2
  subtype     TEXT,                     -- см. словарь §2
  label       TEXT,                     -- 'MH', '2.10', 'TUALETTRUUM', 'Küche'
  label_norm  TEXT,                     -- нормализованная подпись для поиска
  area        REAL,                     -- м²
  length      REAL,                     -- м
  volume      REAL,                     -- м³
  bbox        TEXT,                     -- JSON [x0,y0,x1,y1] в координатах источника
  polygon     TEXT,                     -- JSON [[x,y],...] если есть контур
  page        INTEGER,                  -- для pdf
  source_ref  TEXT,                     -- GUID IFC | handle DXF | 'page:3'
  confidence  REAL DEFAULT 1.0,         -- 1.0 = расчёт, <1 = зрение/OCR
  is_primary  INTEGER DEFAULT 1,        -- 0 = дубль из другого файла того же чертежа
  props       TEXT                      -- JSON: всё прочее
);
CREATE INDEX idx_el_lookup ON elements(drawing_id, type, subtype, floor_id);
CREATE INDEX idx_el_label  ON elements(drawing_id, label_norm);

-- Весь текст с координатами (PDF, DXF, OCR)
CREATE TABLE text_spans (
  id         INTEGER PRIMARY KEY,
  file_id    INTEGER NOT NULL REFERENCES files(id),
  page       INTEGER,
  text       TEXT NOT NULL,
  bbox       TEXT NOT NULL,
  confidence REAL DEFAULT 1.0
);

-- Разобранный шаблон вопроса → следующий такой же бесплатно
CREATE TABLE question_cache (
  id         INTEGER PRIMARY KEY,
  template   TEXT NOT NULL UNIQUE,      -- нормализованный текст вопроса
  fn         TEXT NOT NULL,             -- 'count_objects' | 'room_area' | ...
  args       TEXT NOT NULL              -- JSON
);

-- Результаты прогона стенда
CREATE TABLE eval_results (
  id         INTEGER PRIMARY KEY,
  run_id     TEXT NOT NULL,
  path       TEXT NOT NULL,
  question   TEXT NOT NULL,
  expected   TEXT,
  got        TEXT,
  verdict    TEXT NOT NULL,             -- 'correct' | 'wrong' | 'refused'
  q_type     TEXT,                      -- count | area | text | volume | length
  source     TEXT,
  route      TEXT,                      -- cache | llm | vlm
  cost_usd   REAL,
  latency_ms INTEGER
);
```

---

## 2. Словарь type / subtype — главная договорённость

Менять только все вместе. Парсеры пишут **только эти значения**.

```
room      → bedroom | kitchen | bathroom | wc | sauna | balcony |
            corridor | living | storage | technical | other
apartment → (без subtype)
door      → single | double | sliding | opening     (opening = проём без полотна, НЕ считается)
window    → (без subtype)
fixture   → toilet | sink | bathtub | shower | other
wall      → (без subtype)
slab | column | beam | stair | railing → (без subtype)
```

Нормализация подписей в `subtype` — отдельный словарь:

```
MH, makuuhuone, Zimmer, Stanza, Chambre, bedroom   → bedroom
KH, KPH, PH, PESUH, SH, Bad, Bagno                 → bathroom
WC, Tualett                                        → wc
S, SAUNA                                           → sauna
K, KEITTIÖ, Küche, Cucina, Cuisine                 → kitchen
PARVEKE                                            → balcony
VH                                                 → storage   (гардеробная, НЕ спальня)
Korridor, Couloir, Flur                            → corridor
```

Единицы: **всё в метрах и м²**. Приводим при записи, не при чтении.

---

## 3. Карточка возможностей (`drawings.capabilities`)

Считается при индексации, показывается до первого вопроса.

```json
{
  "counts": "exact",        // exact | estimated | none
  "areas":  "exact",
  "text":   "exact",
  "reason": "DXF с геометрией и подписями",
  "truth_file": "site1000_bldg2082_floor9385.dxf"
}
```

Правила:
- есть DXF или IFC → `counts: exact`, `areas: exact`
- только векторный PDF → `counts: exact`, `areas: none`
- только PNG → `counts: estimated`, `areas: none`
- скан без текста → всё `none`

---

## 4. API

Статика и API с одного Node-процесса. Префикс `/api`.

```
POST   /api/upload              multipart → { jobId }
GET    /api/jobs/:id            SSE: { stage, done, total, message }
GET    /api/drawings            список карточек
GET    /api/drawings/:id        карточка + capabilities + файлы
GET    /api/drawings/:id/plan   геометрия для SVG, §6
POST   /api/ask                 главный, §5
GET    /api/eval/latest         таблица точности для панели
```

---

## 5. `POST /api/ask` — контракт ответа

Запрос:
```json
{ "question": "Kui suur on ruumi „TUALETTRUUM (34-3)\" netopõrandapind?",
  "drawing_id": 12 }
```

Ответ — **всегда одна форма**, различается только `status`:

```json
{
  "status": "answered",
  "answer": 5.58,
  "unit": "m2",
  "confidence": 0.97,
  "route": "cache",
  "evidence": [
    { "kind": "ifc", "ref": "217rX8QDUtdStyz1UVzmsd",
      "label": "TUALETTRUUM 4-3", "floor": "3.korrus" }
  ],
  "cost_usd": 0.0004,
  "latency_ms": 180
}
```

`status` принимает четыре значения:

| status | когда | что ещё в ответе |
|---|---|---|
| `answered` | нашли | `answer`, `unit`, `evidence[]` |
| `estimated` | ответ от зрения или OCR | то же + `confidence` < 1 |
| `needs_clarification` | совпадений несколько | `alternatives[]` |
| `refused` | не можем | `reason`, `missing` |

Отказ всегда говорит, **чего не хватило**:
```json
{ "status": "refused",
  "reason": "В этом файле нет геометрии, только растр",
  "missing": "dxf_or_ifc",
  "searched": "комнаты с подписью MH" }
```

Уточнение:
```json
{ "status": "needs_clarification",
  "question": "Нашлось три комнаты с подписью MH",
  "alternatives": [
    { "id": 441, "label": "MH 2.03", "area": 15.9 },
    { "id": 442, "label": "MH 2.07", "area": 11.3 },
    { "id": 443, "label": "MH 2.11", "area": 12.1 }
  ] }
```

Доказательство (`evidence[]`) — одна из трёх форм:
```json
{ "kind": "ifc",  "ref": "<GUID>", "label": "...", "floor": "..." }
{ "kind": "geom", "ref": "<handle>", "polygon": [[x,y],...], "file_id": 7 }
{ "kind": "page", "ref": "page:3", "bbox": [x0,y0,x1,y1], "file_id": 9 }
```

---

## 6. `GET /api/drawings/:id/plan` — для просмотрщика

Браузер **не видит** ни DXF, ни IFC. Только это, уже в экранных координатах:

```json
{
  "viewBox": [0, 0, 1200, 900],
  "background": "/api/files/31/image",       // png/pdf-рендер, если есть
  "layers": {
    "walls":   [ { "id": 1, "polygon": [[x,y],...] } ],
    "rooms":   [ { "id": 2, "polygon": [[x,y],...], "label": "MH", "area": 15.9 } ],
    "doors":   [ { "id": 3, "bbox": [x0,y0,x1,y1] } ],
    "windows": [ { "id": 4, "bbox": [x0,y0,x1,y1] } ],
    "fixtures":[ { "id": 5, "bbox": [x0,y0,x1,y1], "subtype": "toilet" } ]
  }
}
```

`id` здесь = `elements.id`. Подсветка: фронт получает `evidence[].ref`
или `id` и красит соответствующий элемент. Никакой геометрии фронт не считает.

---

## 7. Функции, которые дёргает LLM (tools)

Только чтение. Числа считает код, модель выбирает функцию и аргументы.

```
count_objects(drawing_id, type, subtype?, floor?)      → число + элементы
room_area(drawing_id, label, floor?)                   → м² + элемент
sum_quantity(drawing_id, type, field, filter?)         → сумма + элементы
list_floors(drawing_id)                                → список этажей
search_text(drawing_id, query)                         → text_spans
```

Каждая возвращает **и число, и список элементов** — из него собирается
`evidence[]`. Функция, вернувшая число без элементов, — ошибка.

---

## Что зафиксировано и не обсуждается после ревью

1. Словарь `type` / `subtype` (§2)
2. Форма ответа `/api/ask` (§5)
3. Единицы: метры и м², приводятся при записи
4. Фронт не парсит исходные файлы (§6)
5. Функция без элементов в ответе — ошибка (§7)
