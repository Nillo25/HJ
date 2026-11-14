import { useEffect, useRef, useState } from 'react';
import { upload, watchJob, type Drawing, type Kind } from '../lib/api';
import { pdfFirstPageToPng } from '../lib/pdfToPng';
import { Badge, Button, Card, CardBody, CardHeader, CardTitle, cn } from './ui';

const KINDS: { kind: Kind; label: string; ext: string; hint: string }[] = [
  { kind: 'dxf', label: 'DXF plan', ext: '.dxf', hint: 'drawn as the real plan; what you ask about is marked on it' },
  { kind: 'ifc', label: 'IFC model', ext: '.ifc', hint: '3D model in the browser, with what you ask about marked' },
  { kind: 'pdf', label: 'PDF sheet', ext: '.pdf', hint: 'text and schedules are read; the lines used are boxed (a scan becomes a PNG)' },
  { kind: 'png', label: 'PNG plan', ext: '.png', hint: 'a raster plan: doors, windows, fixtures and rooms are detected and marked (set the scale for areas)' },
];
const KIND_LABEL: Record<Kind, string> = { dxf: 'DXF plan', ifc: 'IFC model', pdf: 'PDF sheet', png: 'PNG plan' };

type Phase =
  | { name: 'idle' }
  | { name: 'uploading'; file: string; fraction: number }
  | { name: 'parsing'; file: string; stage: string; since: number }
  | { name: 'converting'; file: string }
  | { name: 'error'; message: string };

const STEPS = ['Upload', 'Parse', 'Ready'];

/**
 * Part one of the screen: the file. Pick the kind, drop one file, watch it being parsed, then ask.
 * A new file can replace the current one at any time; a failed replacement keeps the current file.
 */
export function FilePanel({ current, onLoaded }: { current: Drawing | null; onLoaded: (d: Drawing, parsedMs: number) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<Phase>({ name: 'idle' });
  const [replacing, setReplacing] = useState(false);
  const [drag, setDrag] = useState(false);
  const [parsedMs, setParsedMs] = useState<number | null>(null);
  const [converted, setConverted] = useState<{ from: string; pages: number } | null>(null);   // a scanned PDF shown as a PNG
  const [now, setNow] = useState(Date.now());
  const busy = phase.name === 'uploading' || phase.name === 'parsing' || phase.name === 'converting';

  // after a reload: the file was a converted PDF, say so again
  useEffect(() => {
    if (!current) { setConverted(null); return; }
    try {
      const raw = localStorage.getItem(`northline.conv.${current.id}`);
      if (raw && !converted) setConverted(JSON.parse(raw));
    } catch { /* ignore */ }
  }, [current?.id]);                                   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (phase.name !== 'parsing') return;
    const t = setInterval(() => setNow(Date.now()), 200);
    return () => clearInterval(t);
  }, [phase.name]);

  async function send(file: File | undefined, from?: { name: string; pages: number }) {
    if (!file || (busy && !from)) return;
    // the extension decides what the file is (the server checks it once more)
    const byExt = KINDS.find(k => file.name.toLowerCase().endsWith(k.ext));
    if (!byExt) {
      setPhase({ name: 'error', message: `"${file.name}" is not supported. Upload a ${KINDS.map(k => k.label.split(' ')[0]).join(', ')} file.` });
      return;
    }
    const use = byExt.kind;
    const t0 = Date.now();
    try {
      setPhase({ name: 'uploading', file: file.name, fraction: 0 });
      const id = await upload(use, file, f => setPhase({ name: 'uploading', file: file.name, fraction: f }));
      setPhase({ name: 'parsing', file: file.name, stage: 'indexing', since: Date.now() });
      const job = await watchJob(id, j => setPhase({ name: 'parsing', file: file.name, stage: j.stage, since: t0 }));
      if (job.stage === 'failed') throw new Error(job.log.slice(-1)[0]?.trim() || 'The file could not be parsed.');
      const ms = Date.now() - t0;
      setParsedMs(ms);
      const d = job.drawings[0];
      // a picture-only PDF (no text, no drawn shapes) is not vector data: draw page 1 in the browser and treat it as a PNG
      if (d?.kind === 'pdf' && d.vector === false && !from) {
        setPhase({ name: 'converting', file: file.name });
        try {
          const r = await pdfFirstPageToPng(file);
          await send(r.png, { name: file.name, pages: r.pages });
          return;
        } catch (e) {
          setConverted(null);
          setPhase({ name: 'error', message: `This PDF is a picture and could not be converted to an image: ${e instanceof Error ? e.message : e}` });
          return;
        }
      }
      const conv = from ? { from: from.name, pages: from.pages } : null;
      setConverted(conv);
      if (d) { try { conv ? localStorage.setItem(`northline.conv.${d.id}`, JSON.stringify(conv)) : localStorage.removeItem(`northline.conv.${d.id}`); } catch { /* storage may be blocked */ } }
      setPhase({ name: 'idle' });
      setReplacing(false);
      if (d) onLoaded(d, ms);
    } catch (e) {
      setPhase({ name: 'error', message: e instanceof Error ? e.message : String(e) });
    } finally {
      if (input.current) input.current.value = '';
    }
  }

  const showPicker = !current || replacing || busy || phase.name === 'error';
  const step = phase.name === 'uploading' ? 0 : phase.name === 'parsing' ? 1 : current ? 2 : -1;

  return (
    <Card>
      <CardHeader>
        <CardTitle>1 · File</CardTitle>
        {current && !busy && (
          <Button variant="outline" onClick={() => { setReplacing(r => !r); setPhase({ name: 'idle' }); }} className="h-8 px-3 text-xs">
            {replacing ? 'Cancel' : 'Replace file'}
          </Button>
        )}
      </CardHeader>
      <CardBody className="space-y-3">
        {step >= 0 && (
          <ol className="flex items-center gap-2 text-xs" aria-label="Progress">
            {STEPS.map((s, i) => (
              <li key={s} className="flex items-center gap-2">
                <span className={cn('flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-semibold',
                  i < step || (i === 2 && current && !busy) ? 'bg-emerald-600 text-white' : i === step ? 'bg-slate-900 text-white' : 'bg-slate-200 text-slate-500')}>
                  {i < step || (i === 2 && current && !busy) ? '✓' : i + 1}
                </span>
                <span className={i === step ? 'font-medium text-slate-900' : 'text-slate-500'}>{s}</span>
                {i < STEPS.length - 1 && <span className="h-px w-6 bg-slate-300" />}
              </li>
            ))}
          </ol>
        )}

        {phase.name === 'uploading' && (
          <div className="space-y-1">
            <p className="truncate text-sm" title={phase.file}>Uploading {phase.file}…</p>
            <div className="h-2 overflow-hidden rounded bg-slate-200" role="progressbar" aria-valuenow={Math.round(phase.fraction * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full bg-slate-900 transition-all" style={{ width: `${Math.round(phase.fraction * 100)}%` }} />
            </div>
            <p className="text-xs text-slate-500">{Math.round(phase.fraction * 100)}%</p>
          </div>
        )}
        {phase.name === 'converting' && (
          <p className="text-sm" aria-live="polite">
            <b>{phase.file}</b> is a picture without vector data (a scan): converting page 1 to a PNG in your browser…
          </p>
        )}
        {phase.name === 'parsing' && (
          <p className="text-sm" aria-live="polite">
            {phase.stage === 'rendering' ? 'Drawing the sheet' : phase.stage === 'analyzing' ? 'Reading the plan (detecting doors, windows, fixtures, rooms)' : phase.stage === 'queued' ? 'Waiting for a free slot' : 'Parsing the file'}… <span className="tabular-nums text-slate-500">{((now - phase.since) / 1000).toFixed(1)} s</span>
          </p>
        )}

        {phase.name === 'error' && (
          <p className="rounded-md bg-rose-50 p-2 text-sm text-rose-800" role="alert">
            {phase.message}
            {current && <span className="block text-xs text-rose-700">The previous file is still loaded.</span>}
          </p>
        )}

        {current && !showPicker && (
          <div className="space-y-1">
            <p className="truncate text-sm font-medium" title={current.name}>{current.name}</p>
            <p className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
              <Badge tone="blue">{KIND_LABEL[current.kind]}</Badge>
              <span>{current.kind === 'pdf'
                ? `${(current.text_lines ?? 0).toLocaleString('en-US')} text lines read${current.elements ? `, ${current.elements} schedule rows` : ''}`
                : current.kind === 'png' ? (current.raster ? 'read by the floor-plan service' : 'shown as uploaded')
                : current.elements ? `${current.elements.toLocaleString('en-US')} objects parsed` : 'not parsed (no objects)'}</span>
              {parsedMs != null && <span className="tabular-nums">in {(parsedMs / 1000).toFixed(1)} s</span>}
            </p>
            {converted && (
              <div className="space-y-1 text-xs text-amber-900">
                <p>Converted from <b>{converted.from}</b>: the PDF holds only a picture (no text, no vector drawing), so it is handled as an image.</p>
                {converted.pages > 1 && (
                  <p role="alert" className="rounded-md border border-amber-300 bg-amber-50 px-2 py-1.5 font-medium">
                    ⚠ Only page 1 of {converted.pages} was converted. The other {converted.pages - 1} page{converted.pages - 1 === 1 ? ' is' : 's are'} not
                    available: questions are about page 1 only.
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {showPicker && !busy && (
          <>
            <div role="button" tabIndex={0}
              onClick={() => input.current?.click()}
              onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
              onDragOver={e => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={e => { e.preventDefault(); setDrag(false); send(e.dataTransfer.files[0]); }}
              className={cn('flex cursor-pointer flex-col items-center gap-1 rounded-md border-2 border-dashed px-3 py-6 text-center text-sm transition-colors',
                drag ? 'border-slate-900 bg-slate-100' : 'border-slate-300 hover:bg-slate-50')}>
              <span className="font-medium">{current ? 'Drop another file here or click' : 'Drop a file here or click'}</span>
              <span className="text-xs text-slate-500">DXF, IFC, PDF or PNG, one file</span>
            </div>
            <ul className="space-y-0.5 text-xs text-slate-500">
              {KINDS.map(k => <li key={k.kind}><b className="text-slate-700">{k.ext}</b> {k.hint}</li>)}
            </ul>
            <input ref={input} type="file" accept={KINDS.map(k => k.ext).join(',')} className="hidden" onChange={e => send(e.target.files?.[0])} />
          </>
        )}
      </CardBody>
    </Card>
  );
}
