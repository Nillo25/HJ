import { FormEvent, useEffect, useState } from 'react';
import { addRasterObject, ask, undoRaster, clearCache, getDrawing, type Answer, type Drawing, type Part } from './lib/api';
import { AnswerCard } from './components/AnswerCard';
import { EvidenceTable } from './components/EvidenceTable';
import { FilePanel } from './components/FilePanel';
import { IfcViewer } from './components/IfcViewer';
import { PdfViewer } from './components/PdfViewer';
import { PlanViewer } from './components/PlanViewer';
import { RasterTools } from './components/RasterTools';
import { UsageBar } from './components/UsageBar';
import { Button, Card, CardBody, CardHeader, CardTitle, Input } from './components/ui';

const KEY = 'northline.current';

/** what the person typed after «+ Add» and a click -> the object's class (any language they are likely to use) */
const MARK_WORDS: [RegExp, string][] = [
  [/kitchen|кухн|keittiö|kök|küche|\bkk\b/i, 'room_kitchen'], [/bedroom|спальн|makuuhuone|sovrum|schlafzimmer|\bmh\b/i, 'room_bedroom'],
  [/bathroom|washroom|ванная|санузел|kylpyhuone|badrum|badezimmer|\bkh\b/i, 'room_bath'],
  [/balcon|balkon|балкон|parveke|terrace|террас/i, 'room_outdoor'], [/living|гостин|olohuone|vardagsrum|wohnzimmer/i, 'room_living'],
  [/hall|entry|прихож|коридор|eteinen|hall/i, 'room_entry'], [/storage|closet|кладов|гардероб|varasto|förråd/i, 'room_storage'],
  [/sauna|сауна|bastu|lauteet|полок/i, 'sauna_bench'],
  [/window|окн|ikkun|fönster|fenster/i, 'window'], [/door|двер|ovi|dörr|tür/i, 'door'],
  [/toilet|\bwc\b|унитаз|туалет|wc-istuin|toalett/i, 'toilet'], [/sink|basin|раковин|умывальн|мойк|pesuallas|lavoar/i, 'sink'],
  [/shower|душ|suihku|dusch/i, 'shower'], [/bath|ванн|amme|badkar/i, 'bathtub'],
];
const markClassOf = (text: string) => MARK_WORDS.find(([re]) => re.test(text))?.[1] ?? null;

const remember = (id: string | null) => {
  try { id ? localStorage.setItem(KEY, id) : localStorage.removeItem(KEY); } catch { /* storage may be blocked */ }
};

export default function App() {
  const [current, setCurrent] = useState<Drawing | null>(null);
  const [question, setQuestion] = useState('');
  const [asked, setAsked] = useState<string | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [usageKey, setUsageKey] = useState(0);
  const [ifcTab, setIfcTab] = useState<'3d' | 'plan'>('3d');
  const [pick, setPick] = useState<Part | null>(null);       // a part of an area breakdown shown on the plan
  const [scalePts, setScalePts] = useState<[number, number][] | null>(null);   // raster plan: points picked on the plan for the scale
  const [marking, setMarking] = useState(false);   // raster «+ Add»: one click on a missed object, the detector or Haiku says what it is
  const [markBusy, setMarkBusy] = useState(false);
  const [markAt, setMarkAt] = useState<[number, number] | null>(null);   // clicked, waiting for what it is
  // results of «+ Add» and Undo are shown on the answer card, like any other answer
  const say = (what: string, ok: boolean, text: string, cost = 0) => {
    setAsked(what); setPick(null);
    setAnswer(ok ? { status: 'answered', answer: text, unit: 'text', route: cost > 0 ? 'llm' : 'template', cost_usd: cost, evidence: [] }
                 : { status: 'refused', reason: text, missing: 'question', evidence: [] });
  };
  const [planVersion, setPlanVersion] = useState(0);     // bump: the plan changed on the server (a correction), draw it again
  const [filesKey, setFilesKey] = useState(0);           // bump: FilePanel reloads the list of files
  const [notice, setNotice] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);

  // after a reload, come back to the file that was open (uploads live on the server for a day)
  useEffect(() => {
    let id: string | null = null;
    try { id = localStorage.getItem(KEY); } catch { /* ignore */ }
    if (id) getDrawing(id).then(setCurrent).catch(() => remember(null));
  }, []);

  function loaded(d: Drawing) {
    setCurrent(d);
    remember(d.id);
    setQuestion('');
    setAsked(null);
    setAnswer(null);          // a new file starts a new conversation
  }

  // a file uploaded earlier, chosen in the list: open it like after a reload (a new file starts a new conversation)
  async function select(id: string) {
    const d = await getDrawing(id);       // throws when the upload has expired: FilePanel says so
    setPick(null);
    setScalePts(null);
    loaded(d);
  }

  // "Clear cache": every upload, the question cache and the floor-plan service's scenes are deleted on the server
  async function clearAll() {
    if (!confirm('Clear the cache? Every uploaded file, the question cache and the parsed raster plans are deleted. '
      + 'Each file has to be uploaded again, and reading a PNG plan costs a Claude call again.')) return;
    setClearing(true);
    try {
      const r = await clearCache();
      setCurrent(null); remember(null);
      setQuestion(''); setAsked(null); setAnswer(null); setPick(null); setScalePts(null);
      const raster = r.raster == null ? '' : 'error' in r.raster ? ` Floor-plan service not cleared: ${r.raster.error}` : `, ${r.raster.plans} raster plans`;
      setNotice(`Cache cleared: ${r.files} files, ${r.templates} cached questions${raster}.` + (r.kept_busy ? ` ${r.kept_busy} file(s) still being parsed were kept.` : ''));
    } catch (e) {
      setNotice(`The cache was not cleared: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setClearing(false);
      setFilesKey(k => k + 1);
      setUsageKey(k => k + 1);
    }
  }

  // «+ Add»: the click is where; then the person says what it is (a button, or words in the question box)
  useEffect(() => {
    if (!marking || scalePts?.length !== 1) return;
    setMarkAt(scalePts[0]); setMarking(false); setScalePts(null);
    setTimeout(() => document.getElementById('q')?.focus(), 0);
  }, [scalePts, marking]);

  function addMark(cls: string, said: string) {
    if (!current || !markAt) return;
    const [x, y] = markAt;
    setMarkAt(null); setMarkBusy(true);
    addRasterObject(current.id, x, y, cls)
      .then(r => {
        const name = r.cls.replace('room_', '').replace('_', ' ');
        if (r.how === 'room') say(said, true, `This room is now a ${name}${r.count != null ? `. Now ${r.count} on the plan` : ''}. Wrong? Press Undo.`);
        else {
          const by = r.how === 'model' ? 'outline from our detector' : r.how === 'haiku' ? `outline by Haiku, $${r.cost_usd.toFixed(4)}` : 'approximate box';
          say(said, true, `Added: ${name} (${by})${r.count != null ? `. Now ${r.count} on the plan` : ''}. Wrong? Press Undo.`, r.cost_usd);
        }
        setPlanVersion(v => v + 1); setUsageKey(k => k + 1);
      })
      .catch(e => say(said, false, e instanceof Error ? e.message : String(e)))
      .finally(() => setMarkBusy(false));
  }

  useEffect(() => { setMarking(false); setMarkAt(null); }, [current?.id]);

  const markTools = current?.kind === 'png' && current.raster ? (
    <span className="flex items-center gap-1">
      <Button variant={marking ? 'default' : 'outline'} className="h-6 px-2 text-xs" disabled={markBusy || (scalePts !== null && !marking)}
        title="Missed an object? Click this, then click on it: our detector looks again, then Haiku"
        onClick={() => { if (marking || markAt) { setMarking(false); setMarkAt(null); setScalePts(null); } else { setMarking(true); setScalePts([]); } }}>
        {markBusy ? 'Adding…' : marking || markAt ? 'Cancel' : '+ Add'}
      </Button>
      <Button variant="outline" className="h-6 px-2 text-xs" disabled={scalePts !== null}
        title="Take back the last correction of this plan (an added or removed mark)"
        onClick={() => {
          undoRaster(current.id)
            .then(r => { say('Undo', true, r.changed ? 'The last correction is undone.' : 'Nothing to undo on this plan.'); if (r.changed) setPlanVersion(v => v + 1); })
            .catch(e => say('Undo', false, e instanceof Error ? e.message : String(e)));
        }}>Undo</Button>
    </span>
  ) : undefined;
  const markNote = marking
    ? <p className="text-sm font-medium text-slate-800">Click on the plan where the missed object is.</p>
    : markAt
      ? <p className="text-sm font-medium text-slate-800">Marked. Now say in the question box what it is.</p>
    : markBusy ? <p className="text-sm text-slate-600">Adding…</p>
    : null;

  async function run(q: string) {
    if (!current || !q.trim() || loading) return;
    if (markAt) {   // after «+ Add» and a click, the question box says what the object is
      const cls = markClassOf(q);
      setQuestion('');
      if (cls) addMark(cls, q.trim());
      else say(q.trim(), false, `“${q.trim()}” is not something I can add. Say a window, door, toilet, sink, bathtub, shower or sauna; or, clicked inside a room, kitchen, bedroom, bathroom, hall, storage, balcony or living room (the mark is kept). Or press Cancel.`);
      return;
    }
    setAsked(q);
    setQuestion('');          // the box is free for the next question; the asked one stays on the answer card
    setAnswer(null);
    setPick(null);
    setLoading(true);
    try {
      const res = await ask(current.id, q);
      if (res.scene_changed) setPlanVersion(v => v + 1);
      setAnswer(res);
    } catch (e) {
      setAnswer({ status: 'refused', reason: e instanceof Error ? e.message : String(e), missing: 'server' });
    } finally {
      setLoading(false);
      setUsageKey(k => k + 1);
    }
  }
  const onSubmit = (e: FormEvent) => { e.preventDefault(); run(question); };

  // a PNG read by the floor-plan service is drawn like a plan (its scene over the picture); without the service it is just shown
  const hasPlan = current && (current.kind === 'png' ? current.raster : current.kind !== 'pdf');
  const evidence = pick?.evidence ?? answer?.evidence ?? [];

  return (
    <div className="mx-auto max-w-[1500px] px-4 py-5">
      <header className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">North Line</h1>
          <p className="text-sm text-slate-500">Upload a drawing, ask a question, see the answer marked on the drawing.</p>
        </div>
        <div className="flex items-center gap-3">
          <UsageBar refreshKey={usageKey} />
          <Button variant="outline" onClick={clearAll} disabled={clearing || loading} className="h-8 px-3 text-xs"
            title="Delete every uploaded file, the question cache and the parsed raster plans">
            {clearing ? 'Clearing…' : 'Clear cache'}
          </Button>
        </div>
      </header>

      <div className="grid gap-5 lg:grid-cols-[27rem_minmax(0,1fr)]">
        {/* left: the file, the question, the result */}
        <div className="min-w-0 space-y-5">
          <FilePanel current={current} onLoaded={d => { setNotice(null); loaded(d); }} onSelect={async id => { setNotice(null); await select(id); }} refreshKey={filesKey} notice={notice} />

          <Card>
            <CardHeader><CardTitle>2 · Question</CardTitle></CardHeader>
            <CardBody className="space-y-3">
              <form onSubmit={onSubmit} className="flex gap-2">
                <label htmlFor="q" className="sr-only">Question</label>
                <Input id="q" value={question} onChange={e => setQuestion(e.target.value)} disabled={!current}
                  placeholder={!current ? 'Upload a file first' : markAt ? 'What did you mark? e.g. window, toilet' : 'Ask in any language'} autoComplete="off" />
                <Button type="submit" disabled={!current || loading || !question.trim()}>Ask</Button>
              </form>
            </CardBody>
          </Card>

          <AnswerCard question={asked} answer={answer} loading={loading} pick={pick} onPick={setPick}
            onChoose={alt => setAnswer(prev => prev && { ...prev, status: (alt.confidence ?? 1) < 1 ? 'estimated' : 'answered', answer: alt.answer, unit: alt.unit, evidence: alt.evidence, alternatives: undefined, confidence: alt.confidence ?? 1 })} />

          {answer && answer.status !== 'refused' && answer.status !== 'needs_clarification' && <EvidenceTable answer={answer} />}
        </div>

        {/* right: the drawing itself, with what was asked marked on it */}
        <div className="min-w-0 space-y-5 lg:sticky lg:top-4 lg:self-start">
          {current?.kind === 'ifc' && (
            <div role="tablist" aria-label="Model view" className="flex gap-1">
              {([['3d', '3D model'], ['plan', 'Plan by storey']] as const).map(([k, label]) => (
                <button key={k} role="tab" aria-selected={ifcTab === k} onClick={() => setIfcTab(k)}
                  className={ifcTab === k ? 'rounded-md bg-slate-900 px-3 py-1 text-xs font-medium text-white' : 'rounded-md border border-slate-300 px-3 py-1 text-xs text-slate-700 hover:bg-slate-100'}>
                  {label}
                </button>
              ))}
            </div>
          )}
          {/* both stay mounted: switching must not reload a 70 MB model */}
          {current?.kind === 'ifc' && <div className={ifcTab === '3d' ? '' : 'hidden'}><IfcViewer key={current.id} drawingId={current.id} evidence={evidence} /></div>}
          {current?.kind === 'pdf' && <PdfViewer key={current.id} drawingId={current.id} evidence={evidence} />}
          {hasPlan && <div className={current.kind === 'ifc' && ifcTab !== 'plan' ? 'hidden' : ''}><PlanViewer key={`${current.id}-${planVersion}`} drawingId={current.id} evidence={evidence} tools={markTools} note={markNote}
            pick={scalePts && scalePts.length < 2 ? (x, y) => setScalePts(p => (p && p.length < 2 ? [...p, [x, y]] : p)) : undefined} marks={scalePts ?? []} /></div>}
          {current?.kind === 'png' && current.raster && (
            <RasterTools key={`t${current.id}`} marking={marking} drawingId={current.id} points={scalePts} setPoints={setScalePts} onChanged={() => setUsageKey(k => k + 1)} />
          )}
          {current?.kind === 'png' && !current.raster && (
            <Card>
              <CardHeader><CardTitle>Drawing</CardTitle></CardHeader>
              <CardBody className="space-y-2">
                <img src={`/api/drawings/${current.id}/image`} alt="Uploaded raster plan" className="max-h-[70vh] w-full rounded-md border border-slate-200 bg-white object-contain" />
                <p className="text-xs text-slate-500">Raster plans are shown as uploaded. Reading objects from pixels is not supported yet, so questions are refused with a reason.</p>
              </CardBody>
            </Card>
          )}
          {!current && (
            <Card>
              <CardBody className="flex min-h-48 items-center justify-center text-sm text-slate-500">
                Your drawing appears here once the file is parsed.
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
