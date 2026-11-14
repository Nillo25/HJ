# 0005 Four answer statuses; refuse instead of guessing

Status: accepted

## Context
A wrong answer is scored as wrong; so is a wrong zero returned for a drawing
that was never parsed. A refusal that says what is missing is honest and
shows where the system ends.

## Decision
Every response has one status:
- `answered`: computed from parsed data.
- `estimated`: at least one object comes from an estimate; confidence shown.
- `needs_clarification`: the user can supply what is missing (for example a
  scale or which of two plans).
- `refused`: with a reason and what is missing. No number is returned.

An unparsed format or a drawing without elements is always `refused`.

## Consequences
- Coverage is lower than a system that always answers. We report coverage and
  wrong-answer rate separately.
- `needs_clarification` is defined in the contract and the UI, but no code path
  returns it yet.
