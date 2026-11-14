# Everything we know (state at 2026-11-14, evening)

Written for whoever continues: a colleague or a fresh Claude Code session. If something here
disagrees with the code, the code wins; fix this file.

## 1. Event and team

- Junction 2026, challenge by Stoa (Helsinki): "Make Blueprints Talk". Team North Line: Alexy
  (backend and frontend, runs the project), a frontend developer, a tester. cubicasa5k is with
  another person.
- Judging, 25% each: accuracy, how well the solution visualises its evidence so a person can
  verify in seconds, creativity, low inference cost. Inference is measured, so tokens per model
  are shown everywhere.
- Prize: 1000 EUR + work trial at Stoa. Do not talk about it with Alexy.
- Rules: code written specially for the idea before the event starts is disqualifying; knowledge
  transfers, code does not. AI assistants are allowed. Sunday: freeze 08:00, rehearsal, 2-minute
  video (YouTube needs time to process), pitch deck PDF, submit by 10:00, peer review 10:30.
- Stoa engineer at the booth (insights, in order of arrival):
  1. If data is missing, involve the user (ask for a scale, ask which plan) instead of refusing silently.
  2. Judges upload files themselves. One file or one set, then several questions about it.
     There is no database of files on their side.
  3. IFC needs no visualisation, questions only. opening_design: questions only.
     DXF is the case that gets visualised, drawn over SVG and PDF.
  Open questions still unanswered: is a refusal counted as a wrong answer; how inference cost is
  measured (per question, per file, total); which source looks most like real customer files;
  are answers in the dataset checked by hand or generated; do engineers use phones on site.

## 2. Dataset (preview, 0.5 GB; full is 10 GB, never downloaded)

- 1055 files: 374 jpg, 306 png, 123 pdf, 66 dxf, 20 dwg, 14 ifc, 10 svg. 7990 questions on 320 files.
  Same name with different extensions = the same drawing in several formats.
- Question record: `{path, question, answer, unit, type}`; type is count (76%), area (20%), text (3%),
  volume and length (under 1%). One question appears in 2-4 languages (EN FI DE FR IT ET DA):
  dedupe 7990 -> 2335. Scoring: counts exact; area/length/volume within 2%; text free-form.
- Questions exist only for 5 sources. Ignored (no questions): europeana, finna, habs, wikimedia,
  and svg/dwg/jpg files.

| source | files | questions | source of truth | notes |
|---|---|---|---|---|
| cubicasa5k | 241 png | 4270 (53%) | pixels only, needs vision | 12 templates, all count, no area questions at all, so no scale/calibration needed. Six are room labels (sauna, kitchen, MH bedroom, separate WC, bathroom KH/KPH/PH/PESUH/SH, balcony parveke), six are objects (windows, doors, toilets, sinks, shower/bath, floors). Hypothesis: plans print window/door codes ("12x15", "OP 10x21"); test OCR before vision. Not parsed; refused. |
| swiss_dwellings | 65 dxf + 65 png + 65 pdf | 2358 (30%) | DXF | PNG/PDF only for display |
| ifc_models | 4 ifc + 3 pdf + 3 svg | 1049 (13%) | IFC | PDFs are rendered from the IFC |
| ifc_bench | 5 ifc | 249 (3%) | IFC + text report | all text answers |
| opening_design | 55 pdf + 20 dwg + 5 ifc + 1 dxf | 64 (1%) | text layer of vector PDF | 3 files with questions, all about schedules |

## 3. How the system answers

    upload -> Hono API -> (spawn) Python indexer -> SQLite
    question -> ask_cli.py template match (free) -> answer
              \-> miss: model parses the question into a function call (cached by template) -> answer
    ifc_bench question -> ifc_facts.py facts JSON -> one model call that only phrases the facts

- DB schema and API contract: `docs/contract.md` (frozen after review). The parsers store
  everything they find, regardless of the questions; a question is a filter over stored data.
- `answer.py`: functions `count_objects`, `room_area`, `apartment_area`, `count_labelled`,
  `sum_quantity`, `list_floors`, `schedule`. `T` is the template table (regex -> function); it
  lowercases the question, so floor names must be matched case-insensitively. Each function
  returns the value and the rows; evidence is built from the rows (`evidence_of`). A function that
  returns a number without rows is a bug.
- Template path costs about 50 ms (process start). A miss goes to the model: `generateObject` with a
  zod discriminated union (`count_objects`, `room_area`, `sum_quantity`, `list_floors`, `report`,
  `unsupported`), temperature 0, cached in `question_cache` by a template of the question (quoted
  labels and numbers replaced by slots). No key = honest refusal with reason `template`.
- Refusal reasons in use: `parser` (format not parsed), `template` (unknown question, no model),
  `element` (nothing matched), `drawing`, `data`, `source_model` (plan PDF rendered from a model
  that is not in the set, e.g. FZK-Haus).
- Server routes: `GET /api/drawings`, `/api/drawings/:uid`, `/:uid/plan[?floor=id]` (geometry in screen
  coordinates; IFC returns `floors` and `coverage`, PDF returns a `background`), `/:uid/sheet.svg` (rendered
  PDF page), `/:uid/image` (PNG as uploaded), `POST /api/ask {drawing_id: uid, question}`,
  `POST /api/upload` (multipart: `kind` + exactly one file; returns jobId), `GET /api/jobs/:id`,
  `/api/usage`, `/api/eval/latest` (only with `EVAL_DB`).
  Per-upload model (docs/adr/0009): the file goes to `uploads/<uid>/data/<source>/`, the indexer writes
  `uploads/<uid>/index.db`; `uid` = hash(indexer sources + kind + bytes), so the same file is a cache hit and a
  parser change invalidates old indexes. `cache.db` holds only `question_cache`. Env: `PYTHON_BIN`,
  `UPLOADS_DIR`, `CACHE_DB`, `UPLOAD_MAX_MB` (100), `INDEX_TIMEOUT_S` (120), `UPLOAD_TTL_HOURS` (24).
- UI (`web/`): UploadPanel, DrawingList, question field + example chips (`web/src/examples.json`,
  from `data/examples.json`), AnswerCard (status, value, route, tokens, cost, time, "what was
  run"), EvidenceTable, PlanViewer (DXF, IFC per storey, PDF sheet; evidence ids highlight objects, zoom
  to the answer, layer toggles). Proof view: found objects red and numbered like the evidence table, the rest
  dimmed, header "Found: door x 211", per-storey counts for IFC that sum to the answer, a warning for found
  objects with no shape in the file; no hover. Evidence is capped at 1000 rows), UsageBar (questions, share answered without a model, tokens, cost, per-model badges).
- Models: Verda inference, OpenAI-compatible, base URL `https://inference.verda.com/v1`.
  DeepSeek 4.1 Flash 1017 (`deepseek-4-1-flash-1017`, EUR 0.1336 in / 0.5346 out per 1M, cached input
  0.0053, 1M context, image input, optional reasoning) and Kimi K3.1 Pro (EUR 1.79 / 8.95).
  Verda's Client ID/secret are for its cloud management API and are not used. OpenAI and Gemini keys
  also exist; Gemini was earlier planned for cubicasa vision. First real call (DeepSeek, ifc_bench
  question on ac20_arc): 1.1 s, 3772 in / 25 out tokens, about $0.0005. Nothing else measured yet.
- The sandbox that built most of this cannot reach Verda or OpenAI (proxy allow-list), so model
  calls must be tested on the Mac.

## 4. Format knowledge (measured; these rules are why the numbers are what they are)

### DXF (swiss_dwellings)
- Hand-rolled reader (ASCII DXF is alternating code/value lines). Layers: ROOM_OUTLINE, ROOM_TEXT,
  UNIT_OUTLINE, WALL, DOOR, WINDOW, FIXTURE, ANNOTATION. Rooms are ready LWPOLYLINEs; coordinates
  are already in metres, so polygon area is m2 (checked against printed labels to 0.01).
- Windows = WINDOW polylines / 3: 65/65 files. Geometry from proximity clusters (gap 0.05 m); a
  window that does not form a clean 3-cluster is still counted but carries no geometry (a wrong box
  would highlight half the plan).
- Doors = DOOR polylines minus ARCs: 65/65. Arcs alone gave 56% (sliding doors have none, double
  doors have two); clustering by proximity merged neighbouring doors.
- Apartments = UNIT_OUTLINE count. Apartment area = sum of contained labelled indoor rooms (point
  in polygon), not the outline area (it includes walls, overshoots 2-3%). Apartment letter is read
  from the plan text ("Whg. B", "App. A", "Wohnung C", "Appartamento D"), which beats file order.
- "Total area of numbered indoor rooms" = rooms with a label, a number (2.01 ... 2.31), not
  outdoor. Excluded: balconies, loggias, terraces, shafts, voids; unlabelled slivers are the voids.
  The naive sum of all outlines gave 15%; after this fix area went 41% -> 89% overall.
- Fixtures (all on one layer): `group.py` clusters primitives and classifies by bounding-box size
  plus the room it sits in (toilet vs washbasin is decided by the room; bands are deliberately
  wide, ambiguous goes to "other"). Thresholds are tuned on European plans. Toilets, washbasins,
  bathtubs are still off by 1-2 on some files: measure thresholds with `group.histogram()`.
- Known leftovers: area by literal label undercounts ("Zimmer", "Stanza"); some apartments counted
  twice; 73 wrong of 753 deduped questions (9.7%), mostly these.
- Room label dictionary: `labels.py` (Finnish, German, French, Italian, Estonian, English seeds;
  `VH` = storage, not bedroom; `S/SAUNA` = sauna). Unknown labels go to `unknown_labels.txt`
  (git-ignored) for a batch resolution that is not built.

### IFC (ifc_models, ifc_bench)
- Why not ifcopenshell: wheel did not install (glibc); native dependency; we need types, storeys
  and quantities, not geometry. Own STEP reader in `parse_ifc.py`; 61-71 MB files parse in seconds.
  web-ifc was only used for the abandoned viewer spike (Node 61 MB in 511 ms; Y-up geometry).
- Questions are mostly per storey ("how many doors on 3.korrus"). Storey membership comes from
  `IfcRelContainedInSpatialStructure` for elements and `IfcRelAggregates` for spaces. Whole-model
  counts use storey-owned elements only; unowned same-type elements are marked `is_primary=0`
  (stairs 579 -> 15 after this; the 579 were duplicates of the owned ones).
- Walls = IFCWALL + IFCWALLSTANDARDCASE (99 + 747 = 846 matched). Wall quantities: NetSideArea,
  NetVolume, Length. Verified exact on molio "Grundplan": 505.175 m2 / 177.896 m3 / 185.129 m;
  slabs on "Kaelder" NetVolume 277.334. IFC lengths are in the project unit (mm) and are scaled
  to m; areas and volumes are already m2/m3.
- Room area from the space FootPrint contour: 200/200 match, or the Qto set.
- Plumbing: "sanitary fixtures" in `bsc-clinic_clinic-plumbing.ifc` = IfcFlowTerminal (119, exact);
  that model has no architectural elements.
- One PDF name does not derive from the IFC name by a simple rule
  (`kit-fzk-haus_...plan-02-dachgeschoss.pdf`); the FZK-Haus model itself is absent from the set, so
  its questions refuse with `source_model`. `resbim-388_model.ifc` is a 2.5 KB stub, not a bug.
- ifc_bench (text answers): `ifc_facts.py` builds the facts: project/site/building names, software
  (IfcApplication name+version, plus exporter string from the STEP header), schema, levels with
  elevation and height to the next, spaces by level (number, name, area), largest spaces, counts
  by type, door/window types and width range, slab area by level. Door type names come from
  `IFCRELDEFINESBYTYPE` -> `IFCDOORTYPE`; Archicad door width/height attributes are `$`, so width
  and height fall back to the Qto values. Checked on ac20_arc: 5 levels, 82 spaces, 1939.4 m2,
  Keller -3.0 m, 77 doors (67+6+3+1 by type), largest space "Seminarraum (006)" 128.3 m2.
  duplex_arc has no space areas (0 means unknown). Expected answer example: "IFC file generated by
  GRAPHISOFT ARCHICAD-64 20.0.0 GER FULL Windows version. IFC version is 4."
- Status: ifc_models 193 right / 0 wrong / 11 refused (7 counts, 2 volume, 1 length, 1 area).

### Vector PDF (opening_design, US architecture practice)
- pdfminer.six, line-level text with coordinates; rows grouped by baseline; columns by x ranges.
  Sheet A-801: door schedule (38 doors, e.g. numbers 100.A1 ... 130.B) and finish room schedule
  (29 rooms; 2 have area "Redundant" = None). Dimensions like 3' - 0" are normalised to 3'-0" and
  inches. Margin grid markers ("E", "D") left of the table are dropped (`x > 100`).
- Speed is fine: A700 0.5 s, A-701 1.2 s, A-801 1 s. The batch indexer hangs on the large archive
  PDFs in `opening_design/*/Transfer/_Archive` (55 PDFs, up to 25 MB, 22 s each): no per-file limit.
  An earlier claim that A700 was slow was wrong; corrected.
- Done: A-801 sheet, 21 right / 0 wrong. Open: A700 (20 q, 657 door entries), A-701 symbols (3 q),
  one A-801 row question ("four doors of type B2").

### Raster (cubicasa5k)
- Not built. All 4270 questions refuse. Plan: OCR the printed codes first; then a vision model asked
  to enumerate objects with boxes (count = length of the array, computed by code), one cropped
  fragment at a time; cache by file hash. Some PNGs have a vector twin (same name) and could be
  answered exactly; 10 SVGs in the preview were never checked against cubicasa plans.

## 5. Results history

| time (Nov 14) | commit | result |
|---|---|---|
| 11:12 | e6b446e | skeleton: schema, DXF and IFC parsers, eval, API |
| 11:33 | 439ce24 | swiss 41% -> 89% (no balconies, apartment letter, doors = polylines - arcs 65/65) |
| 11:53 | 7690e3a | own STEP reader, footprint areas 200/200, plan PDFs routed to the model, refusal instead of zero |
| 12:05 | 5ecdc18 | PDF schedule parser; A-801 21/0 |
| 13:08-13:28 | 295da45 ad3f16a d7dfacd | evidence, source endpoint, window geometry, template-first CLI, usage, whole web UI |
| 13:40 | 8017cd7 | IFC rework: ifc_models 193/0/11 (was about 54% answered, 6.4% wrong); ifc_facts + report |
Full run before the PDF parser: 767 right, 86 wrong, 1480 refused of 2335 (3.7% wrong). Pitch
framing: answer 60% and be wrong 2%, rather than answer everything and be wrong 30%.

## 6. Decisions that replaced earlier ones (do not resurrect without a reason)

- Angular + Material -> React + Vite + Tailwind (`docs/brief.md` still lists Angular).
- "Do not build upload UI" -> built, because judges upload files.
- ifcopenshell -> own STEP reader (`docs/adr/0003`). web-ifc/three.js IFC viewer -> abandoned: a plan
  cut from 3D came out nearly empty because many elements are not attached to a storey; and
  the organisers said IFC needs no visualisation (`docs/adr/0007`). The spike stays in
  `docs/ifc-viewer-spike/` (first render was tilted because web-ifc output is already Y-up).
- One dataset-wide database -> one small database per upload (decided, not built; see backlog and
  handoff prompt). Routing is manual: the user picks the kind of material.
- No fine-tuning: questions repeat as templates, a tuned model would not carry to real files.
- Statuses: `needs_clarification` is defined and rendered, but no path returns it. This is where
  the "involve the user" insight belongs (scale click, choose between rooms).

## 7. Mistakes already made (so nobody repeats them)

- A python patch script raised before writing the file, a second script half-applied: always run
  patches with an assertion and re-read the result.
- `server/node_modules` installed in the Linux sandbox broke on macOS (esbuild platform binary).
- Provider package version: `@ai-sdk/openai` latest is incompatible with `ai` v4; use `^1`. The
  provider ignored `OPENAI_BASE_URL` until `createOpenAI({ baseURL })` was used.
- A stale `app.db` (built before parser changes) gave a wrong model answer because the facts were
  missing. Add a parser version to the DB.
- `server/.env.example` was never committed (restored in this commit).
- The mock server for generateObject must return the tool_call format.
- Printing the whole IFC `qto` dict flooded the terminal; `qto` is no longer kept in props.
- Playwright (preinstalled in the Linux sandbox) was used for UI checks: the full UI was verified
  once in Chromium; it has not been tested on the Mac.

## 7b. Product-path benchmark (indexer/bench.py) and what it found

`bench.py` uploads a file the way a judge does and asks the dataset questions through the running server; the
language of each question is picked at random. `eval.py` (direct calls, no server) hid every problem below.

| run | wrong | notes |
|---|---|---|
| first (sample of 24 files, 562 q) | 86 of 518 (16.6%) | 68 came from the question cache |
| after the cache fix | 14 of 528 (2.7%) | |
| after parsed calls run in Python, schema fixes | 5 of 529 (0.9%) | ifc_models 193/0, ifc_bench 122 of 241 answered |

- The question cache stored the constants of the first question (a room label, a floor name) and answered all similar
  questions with them. `templatize` also broke its own placeholders (digits inside `{0}`). Fixed: one pass, `abstractArgs`
  turns values taken from the question into slots and does not cache a call that would freeze one. `cache.db` is versioned.
- Verda limits requests per minute (`tenant rpm`): model calls go through a gate (`LLM_RPM`, default 40) and are retried.
- A model-chosen call is executed by `answer.call` (Python), the same code as the templates. `tools.ts` is gone.
- Schema traps: "WC" is a toilet fixture, not a WC room; a room NUMBER in the question (0.22) is the label, the name in
  brackets is not; a compound question ("walls and columns together") is `unsupported`, not half answered.
- "Rooms labelled 'Stanza'": the key counts single-room units printed "App. A" as bedrooms in some plans and not in
  others (identical structure). The file does not say, so the answer is `needs_clarification` with both numbers.
- Apartments: an outline with no room, or under 8 m2, is not an apartment; a second outline with the same letter is ignored.
- Fixtures by bounding box reach a ceiling (threshold search 21 of 34, meaningless values): kept as `estimated`.
- ifc_bench facts (`ifc_facts.py --q`): blocks are chosen by the question words; types, materials, load-bearing per element,
  fire ratings, MEP systems, slabs split by PredefinedType (floor 285 / roof 78 matches the key: 2191.96 m3), site coordinates.
  Where the key says "length not available", we state the footprint-measured figure as an estimate.

## 8. Not yet verified

- The real-model path in several languages; tokens of failed calls; DeepSeek vs Kimi quality.
- ifc_bench answers against the 249 expected texts (harness does not score text).
- Large uploads in the UI (the 74 MB IFC was uploaded through the API, 4.7 s, 310 MB of memory).
- Plan quality on models other than esplanades and molio (ac20_arc, hitos_arc, DXF in the UI after the rewrite).
- Capability card says `areas: none` for schedule sheets although areas are answered.

## 8b. IFC plan (indexer/ifc_geom.py)

- Per element: ObjectPlacement chain -> world matrix; Body (FootPrint for spaces) items:
  IfcExtrudedAreaSolid (rectangle, circle, I-shape, closed polyline; vertical extrusion = exact outline, else hull),
  IfcMappedItem, boolean results (first operand), anything else = convex hull of the reachable points.
  Walls that only have an `Axis` representation get a 15 cm strip along it (thickness is a guess).
- Measured: esplanades 1928 of 1932 drawn (stairs missing), molio 189/189 walls, clinic plumbing 92 of 119
  fixtures; parse 1 to 5 s, up to 420 MB of memory. The UI prints "drawn N of M".
- Storeys come from `owner`; elements without a storey are not primary and not drawn.
- IFC 3D (`web/src/components/IfcViewer.tsx`): the browser downloads the original IFC (`/api/drawings/:id/source`, limit
  `VIEWER_MAX_MB`, 150), parses it with web-ifc (WebAssembly, `web/public/wasm/web-ifc.wasm` is copied by `npm run dev|build`)
  and draws it with three.js. Evidence refs are GlobalIds, mapped to express ids in the browser. Found objects are red with
  numbered sprites, the rest is ghosted, IfcSpace volumes are hidden unless asked about. After an answer the model is cut
  just above the found objects (clipping plane) and the camera looks from above: a real floor plan out of the model.
  web-ifc returns Y-up geometry: do not rotate it again. Measured: 12 MB model about 3 s, 74 MB model under 10 s.
  The 2D "Plan by storey" (footprints, `ifc_geom.py`) stays as the second tab.
- DXF: `render_dxf.py` draws the drawing itself (LINE, LWPOLYLINE/POLYLINE with arcs, ARC, CIRCLE, ELLIPSE, SPLINE, SOLID, HATCH,
  TEXT/MTEXT, INSERT of blocks, entity and layer colours; the eight swiss layers have a fixed style) as `sheet.svg` in metres with Y negated; the plan endpoint returns our shapes in the same space, and only the found
  objects are drawn over the picture. The first 10/20 pair of a HATCH is its elevation point, not a vertex.
- PDF: `render_pdf.py` draws lines, rectangles, curves and text of page 1 as SVG (A-801: 0.8 s, 391 KB);
  schedule row bboxes are in PDF space, Y up.

## 9. Local files that are not in git

`data/opening_test.db`, `data/demo.db`, `data/examples.json`, `data/sync.tgz`, `app.db`,
`unknown_labels.txt`, `server/.env`, `uploads/`. Sandbox scratch (`/tmp/*.db`) is gone with the sandbox.
Stray untracked: root `package-lock.json`, `.idea/` (now ignored).
