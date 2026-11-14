import { unitLabel, type Answer } from '../lib/api';
import { Badge, Card, CardBody, CardHeader, CardTitle } from './ui';

const STATUS = {
  answered: { tone: 'green', text: 'Answered' },
  estimated: { tone: 'amber', text: 'Estimated' },
  refused: { tone: 'red', text: 'Cannot answer' },
  needs_clarification: { tone: 'blue', text: 'Needs your input' },
} as const;

const MISSING: Record<string, string> = {
  parser: 'This file format is not parsed yet.',
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

export function AnswerCard({ question, answer, loading }: { question: string | null; answer: Answer | null; loading: boolean }) {
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

        {answer && answer.status !== 'refused' && (
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
