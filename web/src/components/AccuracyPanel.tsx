import { useEffect, useState } from 'react';
import { Badge, Card, CardBody, CardHeader, CardTitle, Table, Td, Th } from './ui';

type Row = { source: string; type: string; right: number; wrong: number; refused: number; text?: number };
type Summary = {
  generated: string; files: number; questions: number; scored: number; right: number; wrong: number; refused: number;
  wrong_rate: number | null; coverage: number | null; without_model: number; cost_usd: number; cost_per_question_usd: number;
  models: { model: string; calls: number; input_tokens: number; output_tokens: number; cost_usd: number }[];
  rows: Row[];
};

const NAME: Record<string, string> = {
  swiss_dwellings: 'DXF plans', ifc_models: 'IFC models', ifc_bench: 'IFC free-form (text)',
  opening_design: 'PDF schedules', cubicasa5k: 'PNG plans (not supported yet)',
};
const pct = (x: number | null | undefined) => (x == null ? '—' : `${(x * 100).toFixed(1)}%`);

/** The benchmark questions run through this very product: a measurement, not a claim. */
export function AccuracyPanel() {
  const [s, setS] = useState<Summary | null>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    fetch('/api/eval/latest').then(r => (r.ok ? r.json() : null)).then(j => setS(j?.summary ?? null)).catch(() => setS(null));
  }, []);
  if (!s) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Measured accuracy</CardTitle>
        <button onClick={() => setOpen(o => !o)} className="text-xs text-slate-600 underline" aria-expanded={open}>
          {open ? 'Hide details' : 'Show details'}
        </button>
      </CardHeader>
      <CardBody className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Wrong answers" value={pct(s.wrong_rate)} rose hint={`${s.wrong} of ${s.scored} scored`} />
          <Stat label="Answered" value={pct(s.coverage)} hint="the rest is refused with a reason" />
          <Stat label="Without a model" value={pct(s.without_model)} hint="known patterns, 0 tokens" />
          <Stat label="Cost per question" value={`$${s.cost_per_question_usd.toFixed(5)}`} hint={`$${s.cost_usd.toFixed(3)} in total`} />
        </div>
        {open && (
          <>
            <Table>
              <thead>
                <tr><Th>Material</Th><Th>Question</Th><Th className="text-right">Right</Th><Th className="text-right">Wrong</Th><Th className="text-right">Refused</Th></tr>
              </thead>
              <tbody>
                {s.rows.map(r => (
                  <tr key={`${r.source}-${r.type}`}>
                    <Td>{NAME[r.source] ?? r.source}</Td>
                    <Td>{r.type}{r.type === 'text' ? ' (read by hand)' : ''}</Td>
                    <Td className="text-right tabular-nums">{r.right}</Td>
                    <Td className="text-right tabular-nums">{r.wrong ? <Badge tone="red">{r.wrong}</Badge> : 0}</Td>
                    <Td className="text-right tabular-nums">{r.refused}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <p className="text-xs text-slate-500">
              {s.questions} benchmark questions from {s.files} files, {s.generated}. Counts must match exactly, areas and volumes within 2%.
              A refusal is not counted as wrong: a wrong number is worse than none.
              {s.models.map(m => ` ${m.model}: ${m.calls} calls, ${m.input_tokens.toLocaleString('en-US')} in / ${m.output_tokens.toLocaleString('en-US')} out tokens.`).join('')}
            </p>
          </>
        )}
      </CardBody>
    </Card>
  );
}

function Stat({ label, value, hint, rose }: { label: string; value: string; hint?: string; rose?: boolean }) {
  return (
    <div className="rounded-md border border-slate-200 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className={`text-2xl font-semibold tabular-nums ${rose ? 'text-rose-700' : ''}`}>{value}</div>
      {hint && <div className="text-xs text-slate-400">{hint}</div>}
    </div>
  );
}
