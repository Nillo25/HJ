# 0002 Structured formats are the source of truth; images are for display

Status: accepted

## Context
The dataset mixes DXF, IFC, vector PDF and PNG. Same-named DXF, PDF and PNG
files are the same drawing. The IFC plan PDFs were rendered from the IFC.
Some sources (Europeana, Finna, HABS, Wikimedia) have no questions at all.

## Decision
- DXF and IFC hold the data; PNG and PDF next to them are only for display.
- Vector PDF (all PDFs in the dataset) is parsed as text and tables.
- Raster plans (cubicasa5k) are the only case that needs a vision model.
- Sources without questions are ignored by the dataset indexer.
- No fine-tuning: the questions repeat as templates, and a model trained on
  this set would not carry over to real project files.

## Consequences
- Four upload paths with different capabilities (see README).
- Dataset folders ignored by the indexer: europeana, finna, habs, wikimedia;
  SVG, DWG and JPG files.
- Cubicasa5k is the largest question source (4270 of 7990) and the least
  solved. It is tracked separately in `backlog.md`.
