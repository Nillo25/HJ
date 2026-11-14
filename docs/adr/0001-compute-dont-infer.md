# 0001 Compute from parsed data instead of inferring with a model

Status: accepted

## Context
The challenge scores accuracy, verifiability, creativity and inference cost
equally. Questions ask for counts, areas, lengths and volumes of drawing
content. A model that reads a drawing and answers can be wrong without any
way to check it, and it is paid for on every question.

## Decision
Parse each file once into a database. A question is answered by a query over
that data. A model is used only to (a) turn a question into a function call,
(b) phrase text answers, (c) look at raster images.

## Consequences
- Cost grows with the number of files, not the number of questions.
- Every number comes with the rows it was computed from, so it can be checked.
- Accuracy depends on the parsers. Each format needs its own rules, found by
  measuring against the organisers' answers.
- Formats without structured data (PNG) fall back to a vision model and a
  lower, explicitly marked confidence.
