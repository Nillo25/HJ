import { unitLabel, type Alternative, type Answer, type Part } from '../lib/api';
import { Badge, Card, CardBody, CardHeader, CardTitle } from './ui';

const STATUS = {
  answered: { tone: 'green', text: 'Answered' },
  estimated: { tone: 'amber', text: 'Estimated' },
  refused: { tone: 'red', text: 'Cannot answer' },
  needs_clarification: { tone: 'blue', text: 'Needs your input' },
} as const;

const MISSING: Record<string, string> = {
  parser: 'This file format is not parsed yet.',
  source_model: 'Upload the IFC model this plan was made from, together with the plan.',
  template: 'Add a model key to the server, or rephrase as a supported question.',
  element: 'The data has no matching element.',
  drawing: 'The drawing was not found.',
  data: 'The stored data could not be read.',
};

function describe(a: Answer) {
  if (!a.fn) return null;
  const args = Object.entries(a.args ?? {}).filter(([k, v]) => v != null && k !== 'q').map(([k, v]) => `${k}=${JSON.stringify(v)}`);
  return `${a.fn}(${args.join(', ')})`;
}

export function AnswerCard({ question, answer, loading, onChoose, pick, onPick }: { question: string | null; answer: Answer | null; loading: boolean; onChoose?: (alt: Alternative) => void; pick?: Part | null; onPick?: (p: Part | null) => void }) {
  if (!question) return null;
  const s = answer ? STATUS[answer.status] ?? STATUS.refused : null;
  const tokens = (answer?.usage ?? []).reduce((n, u) => n + u.input_tokens + u.output_tokens, 0);
  const isText = typeof answer?.answer === 'string';
  const call = answer ? describe(answer) : null;

  return (
    <Card aria-live="polite" aria-busy={loading}>
      <CardHeader>
        <CardTitle className="truncate" title={question}>{question}</CardTitle>
        {s && <Badge tone={s.tone}>{s.text}</Badge>}
      </CardHeader>
      <CardBody className="space-y-3">
        {loading && <p className="text-sm text-slate-500">Working…</p>}

        {answer?.status === 'needs_clarification' && (
          <div className="space-y-2 rounded-md bg-sky-50 p-3 text-sm text-sky-950">
            <p className="font-medium">{answer.question ?? 'The data allows more than one answer.'}</p>
            <p className="text-xs text-sky-900">The plan does not say which one you mean, so the number is not guessed. Pick one: its objects are shown on the plan.</p>
            <div className="flex flex-wrap gap-2">
              {(answer.alternatives ?? []).map(a => (
                <button key={a.label} onClick={() => onChoose?.(a)}
                  className="rounded-md border border-sky-300 bg-white px-3 py-2 text-left text-sm hover:bg-sky-100">
                  <span className="block font-medium tabular-nums">{a.answer.toLocaleString('en-US')} {unitLabel(a.unit)}</span>
                  <span className="block text-xs text-slate-600">{a.label}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {answer && answer.status !== 'refused' && answer.status !== 'needs_clarification' && (
          <div>
            {isText ? (
              <p className="text-lg font-medium leading-snug">{String(answer.answer)}</p>
            ) : (
              <p className="flex items-baseline gap-2">
                <span className="text-5xl font-semibold tabular-nums tracking-tight">{typeof answer.answer === 'number' ? answer.answer.toLocaleString('en-US') : '—'}</span>
                <span className="text-lg text-slate-500">{unitLabel(answer.unit)}</span>
              </p>
            )}
            {answer.status === 'estimated' && answer.confidence != null && (
              <p className="mt-1 text-sm text-amber-800">Confidence {Math.round(answer.confidence * 100)}%: at least one object comes from an estimate.</p>
            )}
          </div>
        )}

        {answer?.status === 'refused' && (
          <div className="rounded-md bg-amber-50 p-3 text-sm text-amber-950">
            <p className="font-medium">{answer.reason ?? 'The question could not be answered.'}</p>
            {answer.missing && <p className="mt-1 text-amber-900">{MISSING[answer.missing] ?? `Missing: ${answer.missing}`}</p>}
            <p className="mt-2 text-xs text-amber-800">No number is shown on purpose: a wrong zero would look like an answer.</p>
          </div>
        )}

        {answer?.breakdown && answer.breakdown.length > 0 && <Parts answer={answer} pick={pick ?? null} onPick={onPick} />}

        {answer && !loading && <TokenStrip answer={answer} />}

        {answer && (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 border-t border-slate-100 pt-3 text-xs text-slate-600 sm:grid-cols-4">
            <div>
              <dt className="text-slate-400">Route</dt>
              <dd>{answer.route === 'template' || answer.route === 'cache' ? 'known pattern, no model' : answer.route === 'llm' ? 'model parsed the question' : answer.route ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-slate-400">Tokens</dt>
              <dd className="tabular-nums">{tokens.toLocaleString('en-US')}</dd>
            </div>
            <div>
              <dt className="text-slate-400">Cost</dt>
              <dd className="tabular-nums">${(answer.cost_usd ?? 0).toFixed(5)}</dd>
            </div>
            <div>
              <dt className="text-slate-400">Time</dt>
              <dd className="tabular-nums">{answer.latency_ms != null ? `${answer.latency_ms} ms` : '—'}</dd>
            </div>
            {(answer.usage ?? []).map(u => (
              <div key={u.model} className="col-span-2 sm:col-span-4">
                <dt className="text-slate-400">Model</dt>
                <dd className="tabular-nums">{u.model}: {u.input_tokens} in / {u.output_tokens} out</dd>
              </div>
            ))}
            {call && (
              <div className="col-span-2 sm:col-span-4">
                <dt className="text-slate-400">What was run</dt>
                <dd><code className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px]">{call}</code></dd>
              </div>
            )}
          </dl>
        )}
      </CardBody>
    </Card>
  );
}

/** What this one question cost: inference is judged, so it is never hidden. */
function TokenStrip({ answer }: { answer: Answer }) {
  const used = answer.usage ?? [];
  const inT = used.reduce((n, u) => n + u.input_tokens, 0);
  const outT = used.reduce((n, u) => n + u.output_tokens, 0);
  const free = inT + outT === 0;
  return (
    <div className={free ? 'rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-900' : 'rounded-md bg-slate-100 px-3 py-2 text-sm text-slate-800'}
      aria-label="Tokens spent on this question">
      {free ? (
        <><b className="tabular-nums">0 tokens</b> spent: answered from the parsed file, no model was called.</>
      ) : (
        <>
          <b className="tabular-nums">{(inT + outT).toLocaleString('en-US')} tokens</b> spent
          <span className="tabular-nums text-slate-600"> ({inT.toLocaleString('en-US')} in + {outT.toLocaleString('en-US')} out)</span>
          {' · '}<b className="tabular-nums">${(answer.cost_usd ?? 0).toFixed(5)}</b>
          <span className="text-slate-500"> · {used.map(u => u.model).filter((m, i, a) => a.indexOf(m) === i).join(', ')}</span>
        </>
      )}
    </div>
  );
}

/** The area cut into parts: each row is drawn on the plan when picked, so the sum can be checked by eye. */
function Parts({ answer, pick, onPick }: { answer: Answer; pick: Part | null; onPick?: (p: Part | null) => void }) {
  const groups = [...new Set((answer.breakdown ?? []).map(p => p.group))];
  return (
    <div className="space-y-2">
      {answer.note && <p className="text-xs text-slate-500">{answer.note}</p>}
      <div className="max-h-72 overflow-auto rounded-md border border-slate-200">
        {groups.map(g => (
          <div key={g}>
            <div className="sticky top-0 bg-slate-100 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">{g}</div>
            {(answer.breakdown ?? []).filter(p => p.group === g).map(p => {
              const on = pick?.label === p.label && pick?.group === p.group;
              return (
                <button key={g + p.label} onClick={() => onPick?.(on ? null : p)} aria-pressed={on}
                  className={`flex w-full items-center justify-between gap-2 px-2 py-1.5 text-left text-sm ${on ? 'bg-rose-50 text-rose-900' : 'hover:bg-slate-50'}`}>
                  <span className="truncate" title={p.label}>{p.label} <span className="text-xs text-slate-400">· {p.count} room{p.count === 1 ? '' : 's'}</span></span>
                  <span className="shrink-0 tabular-nums font-medium">{p.area.toLocaleString('en-US')} m²</span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
      {pick && <p className="text-xs text-rose-700">Showing “{pick.label}” on the plan. Click it again to show the whole answer.</p>}
    </div>
  );
}
