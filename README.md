# North Line: blueprints that answer

Junction 2026, Stoa challenge "Make Blueprints Talk". Upload a drawing, ask a
question in any language, get an answer with the elements it was computed from.

The idea is **compute, don't infer**. A file is parsed once into a database;
a question becomes a database query. A model only turns the question into a
function call, and reads raster images. A wrong answer costs more than a
refusal, so the system refuses and says what is missing.

## Architecture

    upload ──> Hono API ──spawn──> Python indexer ──> SQLite
                  │                                      ▲
    question ─────┼─> template match (Python, free) ─────┤
                  └─> model parses question (fallback) ──┘
                                │
                  answer + evidence + tokens + cost ──> React UI

| Part | Where | Notes |
|---|---|---|
| Indexer | `indexer/` | Python, stdlib only (plus `pdfminer.six` for PDF). Own DXF, IFC (STEP) and PDF-table parsers. |
| API | `server/` | Hono + better-sqlite3. Upload, ask, plan geometry, usage totals. |
| UI | `web/` | React, Vite, Tailwind, shadcn-style primitives. Builds into `server/public`. |
| Docs | `docs/` | `adr/` decisions, `contract.md` DB schema and API, `backlog.md` what is left. |

## What each upload type does

| Upload | Source of truth | Shown in the UI |
|---|---|---|
| DXF (one file) | DXF | Answer, evidence table, plan drawn from the geometry with highlighted objects |
| IFC (one file) | IFC | Answer, evidence table, plan per storey from element footprints |
| Vector PDF sheet | PDF text and tables | Answer, evidence table, the sheet itself with the rows highlighted |
| PNG plan (cubicasa-like) | not parsed yet | The image and a refusal with the reason |

One file per upload; the user picks the kind. Each upload gets its own index (`uploads/<hash>/index.db`),
so the same file is parsed once. See `docs/adr/0009`.

## Run

    pip install -r indexer/requirements.txt
    cd server && npm i && cp .env.example .env     # add a model key (optional)
    cd ../web && npm i && npm run build            # writes server/public
    cd ../server && PYTHON_BIN=python3.12 npm start   # http://localhost:3000

Needs Python 3.10+ (set `PYTHON_BIN` when the default `python3` is older) and Node 22+. `better-sqlite3` is the only native Node
dependency. Without a model key, known question patterns still work and other
questions are refused with that reason.

Index a dataset folder for evaluation (separate from uploads):

    python indexer/index.py <dataset> --db app.db [--only swiss_dwellings] [--limit N]
    python indexer/eval.py  <dataset> --db app.db [--source swiss_dwellings]

Server settings are in `server/.env`.

## Layout

    indexer/   schema.sql  index.py  eval.py  answer.py  ask_cli.py
               parse_dxf.py  parse_ifc.py  parse_pdf.py  group.py  labels.py
    server/    src/{index,ask,tools,db,usage}.ts
    web/       src/{App.tsx, components/, lib/api.ts}
    docs/      adr/  contract.md  brief.md  backlog.md  ifc-viewer-spike/
    data/      local databases and examples (git-ignored)

## Accuracy

Measured on the preview set, questions deduplicated across languages (7990 → 2335),
using the organisers' scoring: counts exact, area/length/volume within 2%.
The number we quote is the **wrong-answer rate**, not accuracy: a refusal is
not a wrong answer.

| Source | Result (2026-11-14) |
|---|---|
| swiss_dwellings (DXF) | 89% correct (area 87%, count 91%), about 10% wrong |
| ifc_models | 95% answered, 0 wrong, 5% refused |
| opening_design, sheet A-801 | 21 correct, 0 wrong; other sheets refused |
| cubicasa5k, ifc_bench | not parsed, all refused |

Full run before the PDF parser: 767 right, 86 wrong, 1480 refused of 2335 (3.7% wrong).
Re-run `indexer/eval.py` for current numbers; they change with every parser fix.

## Where to start

- What is left and in what order: `docs/backlog.md`.
- Why it is built this way: `docs/adr/`.
- Working rules for this project and dataset facts: `docs/brief.md`.
