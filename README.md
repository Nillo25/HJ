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

One file per upload; the extension decides what it is (DXF, IFC, PDF, PNG). A PDF with no text and no drawn shapes is a scan: the
browser turns its first page into a PNG. Each upload gets its own index (`uploads/<hash>/index.db`),
so the same file is parsed once. See `docs/adr/0009`.

## Run

One command for development (installs npm packages on the first run, finds Python 3.10+, starts the API and the web app):

    npm run dev        # from the repo root; open http://localhost:5173

It reads `server/.env` (optional) and honours `PYTHON_BIN`; Ctrl+C stops both. The steps below are the manual equivalent.

    pip install -r indexer/requirements.txt
    cd server && npm i && cp .env.example .env     # add a model key (optional)
    cd ../web && npm i && npm run build            # writes server/public (and copies web-ifc.wasm)
    cd ../server && npm run build && PYTHON_BIN=python3.12 npm start   # http://localhost:3000

### Docker

    docker build -t northline .
    docker run --rm -p 3000:3000 -v northline-data:/data --env-file server/.env northline
    # or: docker compose up --build

One image: the compiled API, the built web app, Python 3.11 with pdfminer.six. Uploads and the question cache live in the
volume `/data` (`UPLOADS_DIR`, `CACHE_DB`). The model key and prices come from the environment, never from the image
(`LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LLM_PRICE_IN/OUT`, `LLM_RPM`). Other settings:
`UPLOAD_MAX_MB` (100), `INDEX_TIMEOUT_S` (120), `UPLOAD_TTL_HOURS` (24), `VIEWER_MAX_MB` (150), `PYTHON_BIN`.

### Railway

Build from the Dockerfile. Variables: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_PROVIDER`, `LLM_MODEL`, `LLM_PRICE_IN`, `LLM_PRICE_OUT`
(copy them from `server/.env`), `TRUST_PROXY=1`, and `RAILWAY_RUN_UID=0` (a Railway volume is mounted as root, the image runs as `node`).
Add a volume mounted at `/data` so uploads and the question cache survive a deploy. `PORT` is set by Railway.
Healthcheck path: `/api/usage`. A 100 MB IFC needs about 1 GB of memory while it is parsed.

### Limits (flood protection, no login)

| Setting | Default | What it does |
|---|---|---|
| `RATE_ASK_PER_MIN` | 60 | questions per minute per client address (token bucket, burst of the same size) |
| `RATE_UPLOAD_PER_MIN` | 6 | uploads per minute per client address |
| `MAX_PARALLEL_INDEX` | 2 | files parsed at once; the others wait ("queued") |
| `RATE_ASK_PER_DAY` | 50 | questions per day per client address (UTC day); public addresses only |
| `LLM_DAILY_BUDGET` | 20 | spending cap on the model per UTC day, in the currency of `LLM_PRICE_IN/OUT` (set them in EUR); questions that need no model keep working |
| `UPLOADS_MAX_GB` | 5 | disk quota for uploads: the oldest are removed to make room |
| `UPLOAD_MAX_MB` | 100 | file size |

Requests from this machine and from private networks (loopback, 10.x, 172.16-31.x, 192.168.x, which includes Docker's gateway)
are **not** rate limited, so running the code, `indexer/bench.py` or the container on one's own computer is never slowed down
(`RATE_LIMIT_PRIVATE=1` limits them too). Behind a reverse proxy set `TRUST_PROXY=1` (the number of proxies in front): the client address is then read from
the end of `X-Forwarded-For`, where the proxy writes it (the left side can be forged by the client); without it every request looks like the proxy's private address and nothing is limited. A question is at most
2000 characters. `0` switches a limit off. There is no authentication.

Needs Python 3.10+ (set `PYTHON_BIN` when the default `python3` is older) and Node 22+. `better-sqlite3` is the only native Node
dependency. Without a model key, known question patterns still work and other
questions are refused with that reason.

Index a dataset folder for evaluation (separate from uploads):

    python indexer/index.py <dataset> --db /tmp/dataset.db [--only swiss_dwellings] [--limit N]
    python indexer/eval.py  <dataset> --db /tmp/dataset.db [--source swiss_dwellings]

Server settings are in `server/.env`.

## Layout

    indexer/   index.py answer.py ask_cli.py  parse_{dxf,ifc,pdf}.py  render_dxf.py  ifc_geom.py ifc_facts.py
               footprint.py group.py labels.py dict/ (label dictionaries per language)  eval.py bench.py schema.sql
    server/    src/{index,ask,db,usage,py}.ts  test/  accuracy.json (last benchmark, shown in the UI)
    web/       src/{App.tsx, components/, lib/}  (PlanViewer DXF/IFC plan, IfcViewer 3D, PdfViewer, FilePanel ...)
    docs/      adr/  contract.md  brief.md  backlog.md  knowledge.md
    Dockerfile docker-compose.yml

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
