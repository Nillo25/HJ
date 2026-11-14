# 0004 Python indexer, Hono API, React UI

Status: accepted

## Context
The team is stronger in TypeScript, while the parsing work is easier to
iterate on in Python. Indexing is heavy and happens once per file; questions
are light and frequent.

## Decision
- The indexer is a Python program the API starts as a separate process, once
  per upload.
- The API is Hono with better-sqlite3 reading the same SQLite file.
- Known question patterns live in Python (`answer.py`). The API calls
  `ask_cli.py` for them before using a model.
- The UI is React with Tailwind and shadcn-style primitives, served as static
  files by the API.

## Consequences
- Two runtimes to deploy. Keep the data contract in `docs/contract.md` as the
  only coupling between them.
- A question that matches a pattern pays a Python process start (about 50 ms
  measured). If that matters, move the templates into the API.
- The indexer can also run offline over a whole dataset for evaluation.
