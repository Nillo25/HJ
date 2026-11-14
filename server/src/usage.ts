/** Running totals of model usage per model: the jury measures inference cost. */
export type Usage = { model: string; input_tokens: number; output_tokens: number; cost_usd: number };

const totals = new Map<string, Usage & { calls: number }>();
let questions = 0, templateHits = 0;

/**
 * Daily spending cap on the model (UTC day, whole server). Questions that need no model keep working when it is reached.
 * LLM_DAILY_BUDGET_USD, 0 = no cap.
 */
const BUDGET = Number(process.env.LLM_DAILY_BUDGET_USD ?? 20);
let day = new Date().toISOString().slice(0, 10);
let spent = 0;
export class BudgetError extends Error {
  constructor() { super(`The daily model budget ($${BUDGET}) is used up: questions that need the model resume tomorrow (UTC). Questions answered from the parsed file still work.`); }
}
export function assertBudget() {
  const today = new Date().toISOString().slice(0, 10);
  if (today !== day) { day = today; spent = 0; }
  if (BUDGET > 0 && spent >= BUDGET) throw new BudgetError();
}
const failed = new Map<string, number>();       // calls that errored: the provider reports no tokens for them

export function recordFailure(model: string) { failed.set(model, (failed.get(model) ?? 0) + 1); }

export function recordUsage(model: string, input_tokens: number, output_tokens: number, cost_usd: number): Usage {
  const t = totals.get(model) ?? { model, input_tokens: 0, output_tokens: 0, cost_usd: 0, calls: 0 };
  t.input_tokens += input_tokens; t.output_tokens += output_tokens; t.cost_usd += cost_usd; t.calls++;
  spent += cost_usd;
  totals.set(model, t);
  return { model, input_tokens, output_tokens, cost_usd };
}

export function countQuestion(viaTemplate: boolean) { questions++; if (viaTemplate) templateHits++; }

export function usageSnapshot() {
  return {
    questions, template_hits: templateHits,
    models: [...totals.values()].map((m) => ({ ...m, failed_calls: failed.get(m.model) ?? 0 })),
    failed_calls: [...failed.values()].reduce((a, b) => a + b, 0),
  };
}
