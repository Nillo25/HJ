# 0008 Judge by wrong-answer rate on a language-deduplicated question set

Status: accepted

## Context
The preview set has 7990 questions, but the same question appears in 2 to 4
languages, so 2335 are unique. Counts must be exact; area, length and volume
within 2%; text is free-form. Accuracy alone hides whether a miss was a
wrong answer or an honest refusal.

## Decision
`indexer/eval.py` deduplicates by language, applies the organisers' rules, and
reports right, wrong, refused and skipped per source and type. The headline
number is the wrong-answer rate. Coverage is reported next to it. Results are
stored in `eval_results` and readable at `/api/eval/latest`.

## Consequences
- Parser changes are judged by running the harness, not by spot checks.
- Text answers are not scored by the harness (free-form). They are checked by
  hand.
- Whether the organisers count a refusal as wrong is still an open question for
  the booth.
