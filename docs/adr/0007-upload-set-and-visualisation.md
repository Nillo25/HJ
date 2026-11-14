# 0007 Upload is a set of files; only DXF is drawn; IFC and PDF are question-only

Status: accepted

## Context
At the Stoa booth: judges upload files and ask several questions about one
file or set. IFC needs no visualisation; opening_design questions are
questions only; DXF is shown over SVG and PDF.

An IFC viewer was tried first (a spike, three.js + web-ifc, later removed; see git history):
- GlobalId highlighting works, and models open fast (71 MB: 1.4 s for
  geometry in Node, about 520 MB of memory).
- A floor plan cut from the 3D model was not reliable: on the test model many
  elements are not attached to a storey directly, so the plan came out
  nearly empty.

## Decision
- An upload is a set of files grouped by name without extension, stored under
  `uploads/<id>/data/<id>/` and indexed by the same indexer.
- The UI draws a plan only for drawings that include a DXF. For IFC and PDF
  sheets it shows the answer and the evidence table.
- The IFC viewer is not used. `GET /api/drawings/:id/source` and the
  `VIEWER_MAX_MB` cap remain in the API in case it is needed.

## Consequences
- A lone PNG or PDF of a DXF drawing has no sibling to read from. It is
  refused until raster and vector-geometry parsing exist.
- Overlaying the plan on the PNG/PDF needs coordinate registration, which is
  not built. The plan is drawn from DXF geometry as SVG.
- Uploads share one database; two uploads with the same file name collide.
