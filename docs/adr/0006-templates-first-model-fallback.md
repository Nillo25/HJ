# 0006 Known patterns first, model as fallback, usage counted per model

Status: accepted

## Context
Questions in the dataset repeat as templates (about 12 per source, in 2 to 4
languages). Inference cost is judged. Live questions from judges will not
always match.

## Decision
1. `ask_cli.py` matches the question against templates in `answer.py`. A hit
   costs no tokens.
2. A miss goes to a model with a discriminated-union schema (`generateObject`,
   temperature 0) that returns a function call. The result is cached by a
   template of the question (quoted labels and numbers replaced by slots).
3. Without a model key a miss is refused with that reason.
4. Provider is chosen by `LLM_PROVIDER` (openai or google); `LLM_MODEL`,
   `OPENAI_BASE_URL` and `LLM_PRICE_IN/OUT` come from the environment.
5. Every model call is recorded per model (tokens in/out, cost) in
   `server/src/usage.ts` and shown in the UI header and on each answer.

## Consequences
- Most dataset questions need no model at all; the header shows the share.
- Usage totals live in memory and reset when the server restarts. A call that
  fails after spending tokens is not counted.
- Question understanding for new phrasings depends on the model's
  structured-output quality. It has only been tested against a stub so far.
