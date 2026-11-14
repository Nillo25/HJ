# What is left

Question counts are for the preview set (7990 questions, 2335 after language
deduplication). Order is by value for the demo.

## Before the demo

1. **Run with a real model key.** Only a stub was used. Check question parsing
   in several languages; count the tokens of failed calls.
2. **Indexer limits.** Per-file timeout and page cap. The batch indexer
   hangs on the large PDFs in `opening_design/*/Transfer/_Archive` because it
   parses everything. Triage PDFs by size, page count and first-page text.
3. **Deploy.** Python 3.10+ and Node 22+ in one image; `pdfminer.six`
   installed, otherwise PDFs are silently left unparsed.
4. **Submission.** Pitch deck PDF, 2-minute video, demo set, peer review.
   Freeze 08:00 Sunday, submit by 10:00.

## Accuracy

5. **cubicasa5k**, 4270 questions (53% of all), all refused today. Test the
   hypothesis that most plans print window and door codes ("12x15",
   "OP 10x21") with OCR first, then measure a vision model on about 20 plans.
   Currently with another person.
6. **ifc_bench**, 249 text questions. The data is already in the database; a
   model phrases the report.
7. **swiss_dwellings leftovers**: toilets, washbasins and bathtubs off by 1 to
   2 (measure thresholds with `group.histogram()`); area by literal label
   undercounts ("Zimmer", "Stanza"); some apartments counted twice.
8. **ifc_models**: length and volume questions (22) are refused; re-measure
   stair counts after the IFCSTAIR-only change.
9. **opening_design**: A700 schedules (20 questions, 657 door entries), A-701
   symbols (3), and one A-801 row question ("four doors of type B2").

## Product

10. Accuracy panel in the UI. `/api/eval/latest` exists; the page does not.
11. `needs_clarification` flow: defined, never returned.
12. Plan overlay on the PNG/PDF (needs DXF-to-pixel registration).
13. Usage totals persist across restarts; count failed model calls.
14. Capability card says `areas: none` for schedule sheets, but their areas
    are answered.
15. One database for all uploads; names collide across uploads. Give each
    upload its own database or a unique name.

## Unexplained

- `resbim-388_model.ifc` yields no elements in either the indexer or web-ifc.
- The 71 MB IFC was only opened in Node, not in a browser.

## Questions for the organisers

- Is a refusal counted as a wrong answer?
- How is inference cost measured: per question, per file, or in total?
- Which source looks most like real customer files?
