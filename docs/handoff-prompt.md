# Handoff prompt for Claude Code

Paste into Claude Code started in the repo root.

---

You are continuing work on "North Line", our Junction 2026 submission for the Stoa challenge "Make Blueprints Talk". Work in this repo. Answer me in Russian, dry and short. Code comments in English, no fluff. Never print or echo secrets; keys live only in server/.env (git-ignored).

## Goal
Answer natural-language questions about construction drawings / BIM verifiably. Judging: accuracy, visualised verification, creativity, low inference cost (25% each). Judges upload one drawing set at a time: DXF comes as a set with PNG/PDF (visualise the DXF over SVG/PDF); IFC alone, questions only, no visualisation; opening_design PDFs questions only. Inference is measured per model, so token usage must always be visible.

## Principle: compute, don't infer
Parse once into SQLite; questions become queries. The model only (a) parses a question into a function call, (b) phrases text from precomputed facts, (c) reads rasters. Answer statuses: answered / estimated / needs_clarification (defined, never returned yet) / refused (with a reason, never a made-up number). A wrong answer is worse than a refusal: the quoted metric is the wrong-answer rate.

## Layout
- indexer/ (Python, stdlib + pdfminer.six): index.py, parse_dxf.py, parse_ifc.py (hand-rolled STEP reader), parse_pdf.py, group.py, labels.py, answer.py (template answering), ask_cli.py, ifc_facts.py (facts sheet for free-form IFC questions), eval.py, schema.sql.
- server/ (Hono + TypeScript): src/ask.ts is template-first (execFile ask_cli.py, ~50 ms), falls back to an LLM via generateObject (zod discriminated union, temperature 0, question_cache). Provider via env LLM_PROVIDER (openai|google), LLM_MODEL, OPENAI_BASE_URL, LLM_PRICE_IN/OUT. Free-form questions on source ifc_bench go straight to a `report` function: one generateText call over ifc_facts JSON. src/usage.ts records per-model usage (in memory; failed calls not counted yet).
- web/ (Vite 7, React 19, Tailwind v4): upload, drawing list, question field with example chips, AnswerCard, EvidenceTable, PlanViewer (DXF only), UsageBar. Build output goes to server/public.
- docs/: README.md, adr/0001-0008, backlog.md (the source of truth for what is left), ifc-viewer-spike/ (abandoned).
- Dataset: ../junction-dataset-preview (data/<source>/..., questions.jsonl). Keep data/opening_test.db and data/demo.db; /tmp DBs are scratch.

## State (last commits 8017cd7, 83a5ee4)
- eval (python3 indexer/eval.py ../junction-dataset-preview --db <db> --source <s>; index first with indexer/index.py --only <s>): ifc_models 193 right / 0 wrong / 11 refused; swiss_dwellings 89% (about 10% wrong); opening A-801 sheet 21 right / 0 wrong. Do not regress these.
- Scoring: counts exact, area/length/volume within 2%, text answers free-form (not scored by the harness; compare by hand). Language dedupe 7990 -> 2335 questions.
- IFC: quantities in project unit (mm lengths, m2/m3 areas); whole-model counts use storey-owned elements only (is_primary); floor names matched case-insensitively.

## Immediate task: connect the real model and measure it
1. server/.env is set up for Verda (OpenAI-compatible): LLM_PROVIDER=openai, OPENAI_BASE_URL=https://inference.verda.com/v1, LLM_MODEL=deepseek-4-1-flash-1017, LLM_PRICE_IN=0.1336, LLM_PRICE_OUT=0.5346 (EUR per 1M tokens; the panel labels them $). The other available Verda model is Kimi K3.1 Pro (EUR 1.79 in / 8.95 out). Do not read the key value. node_modules was installed on Linux: rm -rf server/node_modules web/node_modules and npm install on this Mac first.
2. Start the server and check one call end to end: a non-template question, and an ifc_bench free-form question (drawing ac20_arc: "What software and ifc standards were used to create this model?"). Check generateObject works with this model (tool calling); if not, switch to mode: 'json'. Check the model does not spend tokens on reasoning by default.
3. Measure DeepSeek vs Kimi on 30-50 questions each: right / wrong / refused, tokens, cost, latency, multi-language questions. Report a short table. Pick by numbers.
4. For ifc_bench (249 text questions) compare answers with the expected text by hand on about 20 questions; fix ifc_facts.py where facts are missing (software string, footprint area, space breakdown, door/window types).
5. Count tokens and cost of failed LLM calls in usage.ts.

## Then (see docs/backlog.md)
Indexer per-file timeout and PDF triage (55 PDFs in opening_design, up to 25 MB, make the batch hang); A700 (20 q) and A-701 (3 q) schedules; swiss leftovers (fixtures, literal-label areas, double apartments); accuracy panel in the UI; needs_clarification flow; per-upload DB and name collisions; PNG/PDF underlay for DXF; deploy image (Python + Node + pdfminer.six); submission deliverables (deck, video, demo set, peer review; freeze 08:00, submit by 10:00). cubicasa5k is with another person: do not touch.

## Rules
Small commits with clear messages. Type-check the server: cd server && npx tsc --noEmit --module nodenext --moduleResolution nodenext --target es2022 --skipLibCheck --strict src/index.ts. Keep code cross-platform. No prize talk, no padding. Offer options when there is a real choice instead of pushing one.
