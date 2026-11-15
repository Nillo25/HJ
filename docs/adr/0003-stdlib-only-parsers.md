# 0003 Own stdlib-only parsers instead of native geometry libraries

Status: accepted

## Context
`ifcopenshell` is the usual IFC library. Its wheel did not install on the
development machine (it needs a newer glibc than the one available), and it
is a native dependency that every deployment target would have to carry.
The questions need entity types, storey membership and quantities, not 3D
geometry.

## Decision
Read DXF and IFC (ISO-10303-21 STEP text) with small readers in
`indexer/parse_dxf.py` and `indexer/parse_ifc.py`. Room area comes from the
space footprint contour or the Qto set. PDF tables use `pdfminer.six`, which
is pure Python.

## Consequences
- The indexer runs anywhere Python 3.10+ runs. The only native dependency in
  the project is `better-sqlite3` on the Node side.
- No geometry kernel. Shapes beyond footprints are not available from IFC.
  If that is ever needed, put `ifcopenshell` behind the same `parse()`
  signature rather than spreading it through the code.
- The parsers carry measured rules (for example windows = WINDOW polylines
  divided by 3, doors = DOOR polylines minus arcs). They are tuned to this
  dataset and need re-measuring on other sources.
