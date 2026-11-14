# North Line (Junction 2026, Stoa "Make Blueprints Talk")

Read `docs/knowledge.md` first: it holds everything learned so far (data facts, parser rules,
measured results, decisions, mistakes). `docs/backlog.md` says what is left, `docs/adr/` why.
`docs/contract.md` is the DB schema and API contract; `docs/brief.md` is the original brief
(some parts are superseded, see the note at its top).

## Working with Alexy
- Answer in Russian, dry and short. No prize talk, no sending him to rest, no comments on his
  self-criticism. Give options when there is a real choice; do not push a single one.
- Do not attribute experience he has not stated. Do not pad.
- Code comments in English, no fluff. Code must run cross-platform (Mac, Linux, Windows).
- Never print, echo or commit secrets. Keys live only in `server/.env` (git-ignored). Do not read
  the key value. If a key is missing, ask him to set it.
- Raster plans (PNG, cubicasa5k) are read by `raster/`, the colleagues' project (YOLO + Claude, FastAPI). Do not edit code in `raster/`
  (it is merged from their repository); the adapter is `server/src/raster.ts`, no pixel reading of our own.
- The product has no dataset database: each upload is `uploads/<uid>/index.db` (docs/adr/0009).
  `data/opening_test.db` and `data/demo.db` are old dev databases (git-ignored). Dataset-wide databases are scratch files for
  `eval.py`; ask before deleting anything that is not obviously a build artefact.

## Principle
Compute, don't infer. Parse once into SQLite; a question becomes a query. A model only turns a
question into a function call, phrases text from precomputed facts, or reads rasters. Four answer
statuses: answered / estimated / needs_clarification / refused. Refuse with a reason instead of
returning a number we cannot stand behind. The quoted metric is the wrong-answer rate.

## Commands
Python must be 3.10+ (macOS system python is 3.9: use `brew install python@3.12` and a venv). The server
starts the indexer with `PYTHON_BIN` (default `python3`): point it at the right interpreter.

    pip install -r indexer/requirements.txt              # pdfminer.six only
    python indexer/index.py <dataset> --db /tmp/x.db --only <source>
    python indexer/eval.py  <dataset> --db /tmp/x.db --source <source> [--show N]
    cd server && PYTHON_BIN=/path/to/python npm run dev   # needs server/.env
    cd server && CACHE_DB=:memory: npm test                                 # question cache tests (node:test via tsx)
    docker build -t northline . && docker run --rm -p 3000:3000 -v northline-data:/data --env-file server/.env northline
    cd web && npm install && npm run build               # writes server/public
    cd server && npx tsc --noEmit --module nodenext --moduleResolution nodenext --target es2022 --skipLibCheck --strict src/index.ts
The dataset is at ~/Downloads/junction-dataset-preview on the Mac (data/<source>/..., questions.jsonl).

## Regression gate (run after any parser or answer change)
ifc_models 193 right / 0 wrong / 11 refused; swiss_dwellings 89% (about 10% wrong);
opening_design sheet A-801: 21 right / 0 wrong. A DB built before a parser change is stale:
reindex it.

## Gotchas
- node_modules are platform-specific: never copy them between Linux and macOS.
- Floor names are matched case-insensitively. IFC lengths are in project units (mm); areas and
  volumes are m2/m3. Whole-model counts use storey-owned elements only (`is_primary`).
- Do not print big dicts (IFC `qto`) to the terminal.
- `pkill -f index.py` also kills the shell that contains it: use a bracket pattern.
- Commit with small messages; end them with the attribution trailers the harness gives you.
