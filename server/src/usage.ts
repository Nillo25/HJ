/** Running totals of model usage per model: the jury measures inference cost. */
export type Usage = { model: string; input_tokens: number; output_tokens: number; cost_usd: number };

const totals = new Map<string, Usage & { calls: number }>();
let questions = 0, templateHits = 0;

export function recordUsage(model: string, input_tokens: number, output_tokens: number, cost_usd: number): Usage {
  const t = totals.get(model) ?? { model, input_tokens: 0, output_tokens: 0, cost_usd: 0, calls: 0 };
  t.input_tokens += input_tokens; t.output_tokens += output_tokens; t.cost_usd += cost_usd; t.calls++;
  totals.set(model, t);
  return { model, input_tokens, output_tokens, cost_usd };
}

export function countQuestion(viaTemplate: boolean) { questions++; if (viaTemplate) templateHits++; }

export function usageSnapshot() {
  return { questions, template_hits: templateHits, models: [...totals.values()] };
}
