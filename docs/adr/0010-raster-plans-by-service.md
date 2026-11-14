# 0010: raster plans are read by a separate service

Date: 2026-11-14. Status: accepted.

## Context
PNG plans (cubicasa5k, scanned PDF pages) need pixels read, not parsed. Colleagues have a working pipeline for exactly this
(`raster/`, repository Nillo25/HJ: YOLO segmentation, a model for the sheet, FastAPI, its own tests and front). It is in Python
with heavy dependencies (torch, ultralytics, OpenCV) and its own model provider (Anthropic).

## Decision
- `raster/` stays a self-contained service as in their repository (moved there unchanged, so their changes can be merged).
- Northline talks to it over HTTP in one file, `server/src/raster.ts`: upload, ask, scene, scale. Their scene and answers are
  turned into ours (`Answer`, `Evidence`, the plan for `PlanViewer`), so the front, limits, budget and counters work as for the
  other formats. `RASTER_URL` empty = the old behaviour (PNG shown, questions refused with a reason).
- Two images: the Northline one stays small (557 MB), the service is 2.5 GB (CPU torch). The service is never published: it
  has no authentication.
- The service keeps scenes on its own disk and is not the system of record: if it loses a scene, the sheet is read again from our copy.

## Consequences
- A PNG upload takes seconds (YOLO) and up to tens of seconds with the model key; the job shows "Reading the plan".
- Its cost (Claude) is reported as `cost_usd` only, without tokens; it is added to the usage totals and the daily budget.
- Answers from a raster plan are never "answered with certainty" unless the service says high confidence: low and medium are estimates.
- Two currencies of accounting stay open: the service prices are in USD, the daily budget is in the unit of `LLM_PRICE_*`.
