import { useEffect, useState } from 'react';
import { autoRasterScale, getRaster, reanalyzeRaster, setRasterScale, type RasterInfo } from '../lib/api';
import { Button, Card, CardBody, CardHeader, CardTitle } from './ui';

const UNITS = ['m', 'cm', 'mm', 'ft', 'in'];

/**
 * A raster plan has no real units: areas are in pixels until the scale is known. The scale is set by two points on
 * the plan and the real distance between them, or read from the sheet by the model (auto). The points are picked on
 * the plan itself (`points` is shared with PlanViewer through the parent).
 */
export function RasterTools({ drawingId, points, setPoints, onChanged }: {
  drawingId: string;
  /** null = not picking; otherwise the points picked so far (0..2) */
  points: [number, number][] | null;
  setPoints: (p: [number, number][] | null) => void;
  onChanged: () => void;
}) {
  const [info, setInfo] = useState<RasterInfo | null>(null);
  const [length, setLength] = useState('');
  const [unit, setUnit] = useState('m');
  const [busy, setBusy] = useState<'save' | 'auto' | 'again' | null>(null);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const load = () => getRaster(drawingId).then(setInfo).catch(() => setInfo(null));
  useEffect(() => { setInfo(null); setMsg(null); setPoints(null); load(); }, [drawingId]);   // eslint-disable-line react-hooks/exhaustive-deps

  async function run<T>(what: 'save' | 'auto' | 'again', f: () => Promise<T>, done: (r: T) => string) {
    setBusy(what); setMsg(null);
    try { const r = await f(); setMsg({ ok: true, text: done(r) }); setPoints(null); setLength(''); await load(); onChanged(); }
    catch (e) { setMsg({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(null); }
  }

  if (!info) return null;
  const ready = points?.length === 2;
  const dist = ready ? Math.hypot(points![0][0] - points![1][0], points![0][1] - points![1][1]) : 0;
  const len = Number(length.replace(',', '.'));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Scale</CardTitle>
        <span className="text-xs text-slate-500">{info.objects} objects, {info.rooms} rooms read</span>
      </CardHeader>
      <CardBody className="space-y-3 text-sm">
        <p>
          {info.px_per_m
            ? <>Scale: <b className="tabular-nums">{info.px_per_m.toFixed(1)} px/m</b> <span className="text-slate-500">({info.scale_source ?? 'set'}): areas are in m².</span></>
            : <span className="text-amber-800">Scale is not set: areas are in px². {info.scale_detail}</span>}
        </p>

        {info.stale && (
          <p className="flex flex-wrap items-center gap-2 rounded-md bg-amber-50 px-2 py-1.5 text-xs text-amber-900">
            This plan was read by an older version of the reader.
            <Button variant="outline" className="h-7 px-2 text-xs" disabled={busy !== null}
              onClick={() => run('again', () => reanalyzeRaster(drawingId), () => 'Read again.')}>
              {busy === 'again' ? 'Reading…' : 'Read again'}
            </Button>
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" className="h-8 px-3 text-xs" disabled={busy !== null}
            onClick={() => { setMsg(null); setPoints(points ? null : []); }}>
            {points ? 'Cancel' : 'Set scale: two points'}
          </Button>
          <Button variant="outline" className="h-8 px-3 text-xs" disabled={busy !== null || points !== null}
            title="The model reads the scale bar or the printed areas on the sheet (one paid call)"
            onClick={() => run('auto', () => autoRasterScale(drawingId), r => r.px_per_m ? `Scale read: ${r.px_per_m.toFixed(1)} px/m.` : r.detail ?? 'The scale was not found.')}>
            {busy === 'auto' ? 'Reading…' : 'Read scale automatically'}
          </Button>
        </div>

        {points && (
          <div className="space-y-2 rounded-md border border-slate-200 p-2">
            <p className="text-xs text-slate-600">
              {points.length < 2 ? `Click on the plan: point ${points.length + 1} of 2 (the ends of a known distance, e.g. a door width or a wall).` : `Distance picked: ${dist.toFixed(0)} px.`}
            </p>
            {ready && (
              <form className="flex flex-wrap items-center gap-2" onSubmit={e => {
                e.preventDefault();
                if (!(len > 0)) return;
                run('save', () => setRasterScale(drawingId, { x1: points![0][0], y1: points![0][1], x2: points![1][0], y2: points![1][1], length: len, unit }),
                  r => `Scale set: ${r.px_per_m.toFixed(1)} px/m.`);
              }}>
                <label className="sr-only" htmlFor="rlen">Real distance</label>
                <input id="rlen" value={length} onChange={e => setLength(e.target.value)} inputMode="decimal" placeholder="real distance"
                  className="h-8 w-28 rounded-md border border-slate-300 px-2 text-sm" />
                <label className="sr-only" htmlFor="runit">Unit</label>
                <select id="runit" value={unit} onChange={e => setUnit(e.target.value)} className="h-8 rounded-md border border-slate-300 px-1 text-sm">
                  {UNITS.map(u => <option key={u}>{u}</option>)}
                </select>
                <Button type="submit" className="h-8 px-3 text-xs" disabled={busy !== null || !(len > 0)}>{busy === 'save' ? 'Saving…' : 'Set scale'}</Button>
                <Button type="button" variant="outline" className="h-8 px-3 text-xs" onClick={() => setPoints([])}>Pick again</Button>
              </form>
            )}
          </div>
        )}

        {msg && <p role={msg.ok ? 'status' : 'alert'} className={msg.ok ? 'text-xs text-emerald-800' : 'rounded-md bg-rose-50 p-2 text-xs text-rose-800'}>{msg.text}</p>}
      </CardBody>
    </Card>
  );
}
