# 0009 One index per upload, manual kind, a plan for every kind

Status: accepted (supersedes the "uploads share one database" consequence of 0007)

## Context
Judges upload one file at a time and ask several questions about it. There is no dataset of
files on the product side. They also want the evidence drawn, not only listed, for every
kind of material.

## Decision
- No dataset-wide database. An upload is parsed into `uploads/<uid>/index.db`. `uid` is a hash
  of the indexer sources, the kind and the file bytes: the same file is never parsed twice,
  and a parser change invalidates every cached index by itself.
- The only shared database is `cache.db` (question template -> parsed function call).
- One file per upload. The user picks the kind (dxf, ifc, pdf, png); it is checked against the
  extension and stored with the upload. Nothing is sniffed.
- Uploads expire after `UPLOAD_TTL_HOURS` (default 24). Parsing has a timeout (`INDEX_TIMEOUT_S`).
- Every kind has a view, driven by the same evidence ids:
  - DXF: SVG from the parsed geometry, no PNG/PDF needed.
  - IFC: a plan per storey from element footprints (`indexer/ifc_geom.py`: placement chains,
    extruded profiles, mapped items, hull fallback, axis strips for walls without a body).
    The UI says how many objects were drawn out of how many.
  - PDF: the vector page rendered to SVG (`indexer/render_pdf.py`), schedule rows as boxes on it.
  - PNG: the image as uploaded, no highlight (vision is on hold).
- A descriptive IFC question that no function covers goes to `report` (one model call over the
  facts sheet). The model returns `enough`; "not enough" becomes a refusal, never an answer.

## Consequences
- The server needs Python 3.10+ (`PYTHON_BIN`) and writes to `uploads/`.
- Elements the file does not let us place are counted but not drawn; the viewer shows the gap.
- Dataset-wide indexing and `eval.py` remain for development; they use scratch databases.
