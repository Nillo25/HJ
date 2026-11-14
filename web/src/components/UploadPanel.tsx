import { useRef, useState } from 'react';
import { upload, watchJob, type Drawing, type Kind } from '../lib/api';
import { Card, CardBody, CardHeader, CardTitle, Button, cn } from './ui';

const KINDS: { kind: Kind; label: string; ext: string; hint: string }[] = [
  { kind: 'dxf', label: 'DXF plan', ext: '.dxf', hint: 'One DXF file. The plan is drawn from its geometry.' },
  { kind: 'ifc', label: 'IFC model', ext: '.ifc', hint: 'One IFC file.' },
  { kind: 'pdf', label: 'PDF sheet', ext: '.pdf', hint: 'One vector PDF sheet (schedules, text).' },
  { kind: 'png', label: 'PNG plan', ext: '.png', hint: 'One raster plan. Not supported yet: questions are refused with a reason.' },
];

export function UploadPanel({ onReady }: { onReady: (drawings: Drawing[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [kind, setKind] = useState<Kind>('dxf');
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cur = KINDS.find(k => k.kind === kind)!;

  async function send(files: File[]) {
    if (!files.length) return;
    setError(null);
    if (files.length > 1) { setError('Upload one file at a time.'); return; }
    if (!files[0].name.toLowerCase().endsWith(cur.ext)) { setError(`"${cur.label}" expects a ${cur.ext} file.`); return; }
    setBusy(true);
    setStatus('Uploading…');
    try {
      const id = await upload(kind, files[0]);
      const job = await watchJob(id, j => setStatus(`Indexing… ${j.stage}`));
      if (job.stage === 'failed') throw new Error(job.log.slice(-1)[0]?.trim() || 'indexing failed');
      setStatus(job.log.includes('cached') ? 'Ready (already indexed)' : 'Ready');
      onReady(job.drawings);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = '';
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload</CardTitle>
      </CardHeader>
      <CardBody className="space-y-3">
        <div role="radiogroup" aria-label="Kind of material" className="grid grid-cols-2 gap-1">
          {KINDS.map(k => (
            <button
              key={k.kind}
              role="radio"
              aria-checked={kind === k.kind}
              onClick={() => setKind(k.kind)}
              disabled={busy}
              className={cn(
                'rounded-md border px-2 py-1.5 text-xs font-medium',
                kind === k.kind ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-300 text-slate-700 hover:bg-slate-100',
              )}
            >
              {k.label}
            </button>
          ))}
        </div>
        <div
          role="button"
          tabIndex={0}
          onClick={() => !busy && input.current?.click()}
          onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && !busy && input.current?.click()}
          onDragOver={e => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => {
            e.preventDefault();
            setDrag(false);
            if (!busy) send([...e.dataTransfer.files]);
          }}
          className={cn(
            'flex cursor-pointer flex-col items-center gap-1 rounded-md border-2 border-dashed px-3 py-6 text-center text-sm transition-colors',
            drag ? 'border-slate-900 bg-slate-100' : 'border-slate-300 hover:bg-slate-50',
          )}
        >
          <span className="font-medium">Drop a {cur.ext} file or click</span>
          <span className="text-xs text-slate-500">{cur.hint}</span>
        </div>
        <input ref={input} type="file" accept={cur.ext} className="hidden" onChange={e => send([...(e.target.files ?? [])])} />
        {busy && <Button variant="outline" disabled className="w-full">Working…</Button>}
        {status && <p className="text-xs text-slate-600" aria-live="polite">{status}</p>}
        {error && <p className="text-xs text-rose-700" role="alert">{error}</p>}
      </CardBody>
    </Card>
  );
}
