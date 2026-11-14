import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import examples from './examples.json';
import { ask, getDrawing, getDrawings, type Answer, type Drawing } from './lib/api';
import { AnswerCard } from './components/AnswerCard';
import { DrawingList } from './components/DrawingList';
import { EvidenceTable } from './components/EvidenceTable';
import { PlanViewer } from './components/PlanViewer';
import { UploadPanel } from './components/UploadPanel';
import { UsageBar } from './components/UsageBar';
import { Button, Card, CardBody, Input } from './components/ui';

type Ex = { q: string; type: string; drawing: string };
const EXAMPLES = examples as Record<string, Ex[]>;

export default function App() {
  const [drawings, setDrawings] = useState<Drawing[]>([]);
  const [current, setCurrent] = useState<Drawing | null>(null);
  const [question, setQuestion] = useState('');
  const [asked, setAsked] = useState<string | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [usageKey, setUsageKey] = useState(0);

  const refresh = useCallback(() => getDrawings().then(setDrawings).catch(() => setDrawings([])), []);
  useEffect(() => { refresh(); }, [refresh]);

  const select = useCallback(async (d: Drawing) => {
    setAnswer(null);
    setAsked(null);
    setCurrent(d);
    try { setCurrent(await getDrawing(d.id)); } catch { /* keep the list row */ }
  }, []);

  const showPlan = current?.kind === 'dxf' || current?.kind === 'pdf' || current?.kind === 'ifc';
  // examples written for this very file first; otherwise generic ones of the same kind (no model names in them)
  const chips = useMemo(() => {
    if (!current) return [];
    const all = [...(EXAMPLES[current.source] ?? []), ...(current.kind === 'ifc' ? EXAMPLES.ifc_bench ?? [] : [])];
    const own = all.filter(e => e.drawing === current.name);
    if (own.length) return own.slice(0, 6);
    return all.filter(e => !/['"“”„«»]/.test(e.q)).slice(0, 6);
  }, [current]);

  async function run(q: string) {
    if (!current || !q.trim()) return;
    setAsked(q);
    setQuestion(q);
    setAnswer(null);
    setLoading(true);
    try {
      setAnswer(await ask(current.id, q));
    } catch (e) {
      setAnswer({ status: 'refused', reason: e instanceof Error ? e.message : String(e), missing: 'server' });
    } finally {
      setLoading(false);
      setUsageKey(k => k + 1);
    }
  }

  const onSubmit = (e: FormEvent) => { e.preventDefault(); run(question); };
  // IFC evidence has kind 'ifc' and is drawn by element id like the others
  const highlight = useMemo(
    () => new Set((answer?.evidence ?? []).filter(e => e.id != null).map(e => e.id as number)),
    [answer],
  );
  const floorHint = useMemo(() => (answer?.evidence ?? []).find(e => e.floor_id != null)?.floor_id ?? null, [answer]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-5">
      <header className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">North Line</h1>
          <p className="text-sm text-slate-500">Ask a question about a drawing. Every answer shows what it was computed from.</p>
        </div>
        <UsageBar refreshKey={usageKey} />
      </header>

      <div className="grid gap-5 lg:grid-cols-[20rem_1fr]">
        <aside className="space-y-5">
          <UploadPanel onReady={async ds => { await refresh(); if (ds[0]) select(ds[0]); }} />
          <DrawingList drawings={drawings} selected={current?.id ?? null} onSelect={select} />
        </aside>

        <main className="min-w-0 space-y-5">
          <Card>
            <CardBody className="space-y-3">
              <form onSubmit={onSubmit} className="flex gap-2">
                <label htmlFor="q" className="sr-only">Question</label>
                <Input id="q" value={question} onChange={e => setQuestion(e.target.value)} disabled={!current}
                  placeholder={current ? `Ask about “${current.name}”` : 'Select or upload a drawing first'} autoComplete="off" />
                <Button type="submit" disabled={!current || loading || !question.trim()}>Ask</Button>
              </form>
              {chips.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {chips.map(c => (
                    <button key={c.q} onClick={() => run(c.q)} disabled={loading}
                      className="max-w-full truncate rounded-full border border-slate-300 bg-white px-3 py-1 text-xs text-slate-700 hover:bg-slate-100 disabled:opacity-50" title={c.q}>
                      {c.q.replace(/^On sheet A-801 of the Xcel Sports Complex drawings, /, '')}
                    </button>
                  ))}
                </div>
              )}
            </CardBody>
          </Card>

          <AnswerCard question={asked} answer={answer} loading={loading} />

          {answer && answer.status !== 'refused' && (
            <div className={showPlan ? 'grid gap-5 xl:grid-cols-2' : ''}>
              <EvidenceTable answer={answer} hover={hover} onHover={setHover} />
              {showPlan && current && <PlanViewer drawingId={current.id} highlight={highlight} hover={hover} floorHint={floorHint} />}
            </div>
          )}
          {current?.kind === 'png' && (
            <Card>
              <CardBody className="space-y-2">
                <img src={`/api/drawings/${current.id}/image`} alt="Uploaded raster plan" className="max-h-[32rem] w-full rounded-md border border-slate-200 bg-white object-contain" />
                <p className="text-xs text-slate-500">Raster plans are shown as uploaded. Reading objects from pixels is not supported yet, so questions are refused with a reason.</p>
              </CardBody>
            </Card>
          )}
          {!answer && showPlan && current && <PlanViewer drawingId={current.id} highlight={highlight} hover={hover} floorHint={floorHint} />}
        </main>
      </div>
    </div>
  );
}
