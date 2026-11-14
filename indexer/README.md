# Indexer and evaluation harness

Python 3.10+, standard library plus `pdfminer.six` (pure Python). No ifcopenshell, no ezdxf, no
pymupdf: DXF and IFC (STEP) are read by our own parsers (`docs/adr/0003`).

    pip install -r requirements.txt
    python index.py <dataset> --db /tmp/x.db --only swiss_dwellings    # or ifc_models, ifc_bench, opening_design
    python eval.py  <dataset> --db /tmp/x.db --source swiss_dwellings --show 12
    python ask_cli.py --db /tmp/x.db --drawing NAME --question "..."
    python ifc_facts.py --db /tmp/x.db --drawing ac20_arc               # facts for free-form IFC questions

`eval.py` needs a DB indexed first. `--api http://localhost:3000/api/ask` runs the same questions over
HTTP. Results: right / wrong / refused per source and type; language-deduplicated (7990 -> 2335).

| file | what |
|---|---|
| `schema.sql` | tables from `docs/contract.md` |
| `index.py` | walks a dataset or an upload folder `<root>/data/<source>/<files>`, groups files by stem, parses, writes |
| `parse_dxf.py` | swiss_dwellings DXF: rooms, apartments, windows, doors, walls, fixtures |
| `group.py` | clustering of DXF primitives, fixture classification, size histograms |
| `parse_ifc.py` | STEP reader, storeys, elements, quantities, building row with header and application |
| `parse_pdf.py` | vector PDF schedules (door schedule, finish room schedule) via pdfminer |
| `labels.py` | room label normalisation and seed dictionary |
| `answer.py` | functions over the DB and the question templates `T` |
| `ask_cli.py` | one question to JSON; the server calls it first |
| `ifc_facts.py` | facts sheet of one IFC model for the model-phrased report |
| `eval.py` | evaluation harness |

Rules per format and measured results: `docs/knowledge.md`. Known gaps: no per-file time or page limit
(large archive PDFs hang a batch run), no parser version in the DB, raster (PNG) not parsed.
