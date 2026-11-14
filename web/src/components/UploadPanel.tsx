import { useRef, useState } from 'react';
import { upload, watchJob, type Drawing } from '../lib/api';
import { Card, CardBody, CardHeader, CardTitle, Button, cn } from './ui';

const ACCEPT = '.dxf,.ifc,.pdf,.png,.svg';

export function UploadPanel({ onReady }: { onReady: (drawings: Drawing[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function send(files: File[]) {
    if (!files.length) return;
    setBusy(true);
    setError(null);
    setStatus(`Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`);
    try {
      const id = await upload(files);
      const job = await watchJob(id, j => setStatus(`Indexing… ${j.stage}`));
      if (job.stage === 'failed') throw new Error(job.log.slice(-3).join(' ').trim() || 'indexing failed');
      setStatus(`Ready: ${job.drawings.length} drawing${job.drawings.length === 1 ? '' : 's'}`);
      onReady(job.drawings);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload</CardTitle>
      </CardHeader>
      <CardBody className="space-y-3">
        <div
          role="button"
          tabIndex={0}
          onClick={() => input.current?.click()}
          onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
          onDragOver={e => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => {
            e.preventDefault();
            setDrag(false);
            send([...e.dataTransfer.files]);
          }}
          className={cn(
            'flex cursor-pointer flex-col items-center gap-1 rounded-md border-2 border-dashed px-3 py-6 text-center text-sm transition-colors',
            drag ? 'border-slate-900 bg-slate-100' : 'border-slate-300 hover:bg-slate-50',
          )}
        >
          <span className="font-medium">Drop files here or click</span>
          <span className="text-xs text-slate-500">DXF + PNG/PDF with the same name, IFC, or PDF. One drawing = one set of files.</span>
        </div>
        <input ref={input} type="file" multiple accept={ACCEPT} className="hidden" onChange={e => send([...(e.target.files ?? [])])} />
        {busy && <Button variant="outline" disabled className="w-full">Working…</Button>}
        {status && <p className="text-xs text-slate-600" aria-live="polite">{status}</p>}
        {error && <p className="text-xs text-rose-700" role="alert">{error}</p>}
      </CardBody>
    </Card>
  );
}
