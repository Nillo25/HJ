import { useEffect, useMemo, useRef, useState } from 'react';
import * as pdfjs from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import type { Evidence } from '../lib/api';
import { Card, CardBody, CardHeader, CardTitle, cn } from './ui';

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

const FOUND = '#e11d48';
type Box = { id: number; n: number; page: number; rect: [number, number, number, number] };   // rect in PDF points, y up

/**
 * The PDF itself, drawn by the browser (vector, images, fonts as in any viewer). What the answer was found in
 * is boxed in red and numbered like the evidence table; pages without a found line stay as they are.
 */
export function PdfViewer({ drawingId, evidence }: { drawingId: string; evidence: Evidence[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const doc = useRef<pdfjs.PDFDocumentProxy | null>(null);
  const loading = useRef<ReturnType<typeof pdfjs.getDocument> | null>(null);
  const task = useRef<pdfjs.RenderTask | null>(null);
  const [pages, setPages] = useState(0);
  const [page, setPage] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [size, setSize] = useState<{ w: number; h: number; scale: number } | null>(null);
  const [viewport, setViewport] = useState<pdfjs.PageViewport | null>(null);
  const [state, setState] = useState<'load' | 'ready' | 'error'>('load');
  const [error, setError] = useState<string | null>(null);
  const pendingFocus = useRef(false);               // after a new answer: zoom to what was found, then scroll to it

  const boxes = useMemo<Box[]>(() => {
    const out: Box[] = [];
    evidence.forEach((e, i) => {
      if (!e.bbox || e.page == null) return;
      try {
        const b = JSON.parse(e.bbox) as number[];
        if (b.length === 4) out.push({ id: e.id ?? i, n: i + 1, page: e.page, rect: [b[0], b[1], b[2], b[3]] });
      } catch { /* not a box */ }
    });
    return out;
  }, [evidence]);

  // load the document once per file
  useEffect(() => {
    let dead = false;
    setState('load'); setError(null); setPage(1); setPages(0);
    (async () => {
      try {
        const res = await fetch(`/api/drawings/${drawingId}/source`);
        if (!res.ok) throw new Error(((await res.json().catch(() => null)) as { error?: string } | null)?.error ?? `${res.status} ${res.statusText}`);
        const lt = pdfjs.getDocument({ data: new Uint8Array(await res.arrayBuffer()) });
        loading.current = lt;
        const d = await lt.promise;
        if (dead) { lt.destroy(); return; }
        doc.current = d;
        setPages(d.numPages);
        setState('ready');
      } catch (e) {
        if (!dead) { setError(e instanceof Error ? e.message : String(e)); setState('error'); }
      }
    })();
    return () => { dead = true; task.current?.cancel(); loading.current?.destroy(); loading.current = null; doc.current = null; };
  }, [drawingId]);

  // an answer on another page: go there
  useEffect(() => {
    if (!boxes.length) return;
    pendingFocus.current = true;
    if (!boxes.some(b => b.page === page)) setPage(boxes[0].page);
    else setZoom(z => z + 1e-6);                      // same page: re-run the drawing effect so it can focus
  }, [boxes]);                                        // eslint-disable-line react-hooks/exhaustive-deps

  // draw the page
  useEffect(() => {
    const d = doc.current, cv = canvas.current, sc = scroller.current;
    if (state !== 'ready' || !d || !cv || !sc) return;
    let dead = false;
    (async () => {
      const pg = await d.getPage(Math.min(Math.max(page, 1), d.numPages));
      const base = pg.getViewport({ scale: 1 });
      const fit = (sc.clientWidth - 2) / base.width;                  // fit the width, then the user's zoom
      const mine = boxes.filter(b => b.page === page);
      let z = zoom;
      if (pendingFocus.current && mine.length) {
        const x0 = Math.min(...mine.map(b => b.rect[0])), x1 = Math.max(...mine.map(b => b.rect[2]));
        const want = Math.min(6, Math.max(1, (sc.clientWidth * 0.55) / (Math.max(x1 - x0, 40) * fit)));
        if (Math.abs(want - zoom) > 0.05) { setZoom(want); return; }   // zoom first, the next pass draws and scrolls
        z = zoom;
      }
      const scale = fit * z;
      const vp = pg.getViewport({ scale });
      const dpr = Math.min(devicePixelRatio || 1, 2);
      cv.width = Math.floor(vp.width * dpr); cv.height = Math.floor(vp.height * dpr);
      cv.style.width = `${vp.width}px`; cv.style.height = `${vp.height}px`;
      task.current?.cancel();
      const t = pg.render({ canvasContext: cv.getContext('2d')!, viewport: vp, transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined } as never);
      task.current = t;
      try { await t.promise; } catch { return; }
      if (dead) return;
      setViewport(vp); setSize({ w: vp.width, h: vp.height, scale });
      if (pendingFocus.current && mine.length) {
        pendingFocus.current = false;
        const px = (x: number, y: number) => [x * vp.scale, vp.height - y * vp.scale];
        const pts = mine.flatMap(b => [px(b.rect[0], b.rect[1]), px(b.rect[2], b.rect[3])]);
        const cx = (Math.min(...pts.map(p => p[0])) + Math.max(...pts.map(p => p[0]))) / 2;
        const cy = (Math.min(...pts.map(p => p[1])) + Math.max(...pts.map(p => p[1]))) / 2;
        sc.scrollTo({ left: Math.max(cx - sc.clientWidth / 2, 0), top: Math.max(cy - sc.clientHeight / 2, 0) });
      }
    })();
    return () => { dead = true; };
  }, [state, page, zoom, drawingId]);

  const here = boxes.filter(b => b.page === page);
  const r = 11;                                       // screen pixels: the badge does not grow with the zoom

  return (
    <Card>
      <CardHeader>
        <CardTitle>{boxes.length ? `Found: ${boxes.length} line${boxes.length === 1 ? '' : 's'} in the PDF` : 'PDF'}</CardTitle>
        <span className="flex items-center gap-1 text-xs">
          <button disabled={page <= 1} onClick={() => setPage(p => p - 1)} className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 disabled:opacity-40" aria-label="Previous page">‹</button>
          <span className="tabular-nums text-slate-600">page {page} / {pages || '…'}</span>
          <button disabled={page >= pages} onClick={() => setPage(p => p + 1)} className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 disabled:opacity-40" aria-label="Next page">›</button>
          <span className="mx-1 text-slate-300">|</span>
          <button onClick={() => setZoom(z => Math.max(0.5, +(z / 1.25).toFixed(2)))} className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100" aria-label="Zoom out">−</button>
          <button onClick={() => setZoom(1)} className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 tabular-nums">{Math.round(zoom * 100)}%</button>
          <button onClick={() => setZoom(z => Math.min(8, +(z * 1.25).toFixed(2)))} className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100" aria-label="Zoom in">+</button>
        </span>
      </CardHeader>
      <CardBody className="space-y-2">
        <div ref={scroller} className="relative h-[68vh] min-h-[22rem] overflow-auto rounded-md border border-slate-200 bg-slate-100">
          {state === 'error' && <p className="p-4 text-sm text-rose-800" role="alert">The PDF cannot be shown: {error}</p>}
          {state === 'load' && <p className="p-4 text-sm text-slate-500">Loading the PDF…</p>}
          <div className={cn('relative inline-block bg-white', state !== 'ready' && 'hidden')}>
            <canvas ref={canvas} className="block" />
            {size && viewport && (
              <svg className="pointer-events-none absolute left-0 top-0" width={size.w} height={size.h} viewBox={`0 0 ${size.w} ${size.h}`}>
                {here.map(b => {
                  // The text boxes come from pdfminer already in the orientation the page is shown in (y up, rotation applied,
                  // e.g. /Rotate 90): they map to the screen by scale alone. pdf.js's own matrix expects the unrotated page.
                  const px = (x: number, y: number) => [x * viewport.scale, viewport.height - y * viewport.scale];
                  const [[x0, y0], [x1, y1]] = [px(b.rect[0], b.rect[1]), px(b.rect[2], b.rect[3])];
                  const x = Math.min(x0, x1), y = Math.min(y0, y1), w = Math.abs(x1 - x0), h = Math.abs(y1 - y0);
                  return (
                    <g key={`${b.id}-${b.n}`}>
                      <rect x={x - 2} y={y - 2} width={w + 4} height={h + 4} fill="rgba(225,29,72,0.22)" stroke={FOUND} strokeWidth={2} rx={2} />
                      <circle cx={x - 2} cy={y + h / 2} r={r} fill={FOUND} stroke="#fff" strokeWidth={1.5} />
                      <text x={x - 2} y={y + h / 2 + r * 0.36} fontSize={r * 1.05} fontWeight={700} fill="#fff" textAnchor="middle" fontFamily="system-ui, sans-serif">{b.n}</text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        </div>
        <p className="text-xs text-slate-500">
          {boxes.length > 0
            ? <span className="text-rose-700">Red boxes are the lines of the PDF the answer comes from, numbered as in the table{boxes.some(b => b.page !== page) ? '; others are on other pages' : ''}.</span>
            : 'Ask a question: the lines it is answered from are boxed here. Use + and − to zoom, scroll to move.'}
        </p>
      </CardBody>
    </Card>
  );
}
