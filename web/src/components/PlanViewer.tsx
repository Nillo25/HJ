import { useEffect, useMemo, useState } from 'react';
import { getPlan, type Plan } from '../lib/api';
import { Card, CardBody, CardHeader, CardTitle, cn } from './ui';

const STYLE: Record<string, { fill: string; stroke: string; width: number; label: string }> = {
  rooms: { fill: '#eef2f7', stroke: '#cbd5e1', width: 1, label: 'Rooms' },
  apartments: { fill: 'none', stroke: '#94a3b8', width: 1.5, label: 'Apartments' },
  walls: { fill: '#475569', stroke: '#334155', width: 1, label: 'Walls' },
  doors: { fill: '#fde68a', stroke: '#b45309', width: 1.5, label: 'Doors' },
  windows: { fill: '#bae6fd', stroke: '#0284c7', width: 1.5, label: 'Windows' },
  fixtures: { fill: '#ddd6fe', stroke: '#6d28d9', width: 1.5, label: 'Fixtures' },
  rows: { fill: 'transparent', stroke: 'none', width: 1, label: 'Schedule rows' },
  slabs: { fill: '#f8fafc', stroke: '#e2e8f0', width: 0.5, label: 'Slabs' },
  columns: { fill: '#64748b', stroke: '#334155', width: 1, label: 'Columns' },
  beams: { fill: 'none', stroke: '#94a3b8', width: 0.8, label: 'Beams' },
  stairs: { fill: '#e2e8f0', stroke: '#64748b', width: 1, label: 'Stairs' },
  railings: { fill: 'none', stroke: '#94a3b8', width: 0.8, label: 'Railings' },
  furniture: { fill: '#f1f5f9', stroke: '#94a3b8', width: 0.8, label: 'Furniture' },
};
const ORDER = ['slabs', 'rooms', 'apartments', 'walls', 'columns', 'beams', 'stairs', 'railings', 'furniture', 'windows', 'doors', 'fixtures', 'rows'];

/** Draws the parsed plan and marks the elements the answer was computed from. */
export function PlanViewer({ drawingId, highlight, hover, floorHint }: { drawingId: string; highlight: Set<number>; hover: number | null; floorHint?: number | null }) {
  const [plan, setPlan] = useState<Plan | null | undefined>(undefined);
  const [floor, setFloor] = useState<number | null>(null);
  const [off, setOff] = useState<Set<string>>(new Set());
  const [fit, setFit] = useState(true);       // true = zoom to the objects the answer used

  useEffect(() => { setFloor(null); }, [drawingId]);
  useEffect(() => { if (floorHint != null) setFloor(floorHint); }, [floorHint]);   // follow the answer to its storey
  useEffect(() => {
    getPlan(drawingId, floor).then(setPlan).catch(() => setPlan(null));
  }, [drawingId, floor]);

  const layers = useMemo(() => ORDER.filter(k => plan?.layers[k]?.length), [plan]);
  const onFloor = useMemo(
    () => (plan ? Object.values(plan.layers).reduce((n, l) => n + l.filter(it => highlight.has(it.id)).length, 0) : 0),
    [plan, highlight],
  );

  // bounding box of everything the answer used, so a small answer on a big sheet is readable
  const focus = useMemo(() => {
    if (!plan || !highlight.size) return null;
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const k of Object.keys(plan.layers)) {
      for (const it of plan.layers[k]) {
        if (!highlight.has(it.id)) continue;
        const pts = it.polygon ?? (it.bbox ? [[it.bbox[0], it.bbox[1]], [it.bbox[2], it.bbox[3]]] : []);
        for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
      }
    }
    if (!isFinite(x0)) return null;
    const pad = Math.max(x1 - x0, y1 - y0) * 0.08 + 10;
    return [x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad];
  }, [plan, highlight]);
  if (plan === undefined) return <Card><CardBody className="text-sm text-slate-500">Loading plan…</CardBody></Card>;
  if (!plan) return null;

  const [, , w, h] = plan.viewBox;
  const toggle = (k: string) => setOff(s => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Plan</CardTitle>
        <span className="flex items-center gap-2 text-xs text-slate-500">
          {highlight.size
            ? (onFloor === highlight.size ? `${highlight.size} highlighted` : `${onFloor} highlighted here, ${highlight.size} in total`)
            : 'ask a question to highlight objects'}
          {focus && (
            <button onClick={() => setFit(f => !f)} className="rounded-md border border-slate-300 px-2 py-0.5 text-slate-700 hover:bg-slate-100">
              {fit ? 'Show whole plan' : 'Zoom to answer'}
            </button>
          )}
        </span>
      </CardHeader>
      <CardBody className="space-y-3">
        {plan.floors && plan.floors.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
            <label htmlFor="floor">Storey</label>
            <select id="floor" value={plan.floor ?? ''} onChange={e => setFloor(Number(e.target.value))}
              className="rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-800">
              {plan.floors.map(f => <option key={f.id} value={f.id}>{f.name || `#${f.id}`} ({f.n})</option>)}
            </select>
            {plan.coverage && (
              <span title="Objects whose shape could be read from the file. The rest are counted but not drawn.">
                drawn {plan.coverage.drawn} of {plan.coverage.total} objects
              </span>
            )}
          </div>
        )}
        <svg viewBox={focus && fit ? focus.join(' ') : `0 0 ${w} ${h}`} className="max-h-[32rem] w-full rounded-md border border-slate-200 bg-white" role="img" aria-label="Floor plan with highlighted objects">
          {plan.background && <image href={plan.background} x={0} y={0} width={w} height={h} />}
          {layers.filter(k => !off.has(k)).map(k => (
            <g key={k}>
              {plan.layers[k].map(it => {
                const on = highlight.has(it.id);
                const hov = hover === it.id;
                const st = STYLE[k];
                const common = {
                  fill: on ? 'rgba(225,29,72,0.45)' : hov ? 'rgba(14,165,233,0.45)' : st.fill,
                  stroke: on ? '#be123c' : hov ? '#0369a1' : st.stroke,
                  strokeWidth: on || hov ? st.width + 2 : st.width,
                };
                if (it.polygon?.length)
                  return <polygon key={it.id} points={it.polygon.map(p => p.join(',')).join(' ')} {...common}><title>{it.label ?? k}</title></polygon>;
                if (it.bbox)
                  return <rect key={it.id} x={it.bbox[0]} y={it.bbox[1]} width={it.bbox[2] - it.bbox[0]} height={it.bbox[3] - it.bbox[1]} {...common}><title>{it.label ?? k}</title></rect>;
                return null;
              })}
            </g>
          ))}
        </svg>
        <div className="flex flex-wrap gap-2 text-xs">
          {layers.map(k => (
            <button
              key={k}
              onClick={() => toggle(k)}
              aria-pressed={!off.has(k)}
              className={cn('inline-flex items-center gap-1.5 rounded-md border px-2 py-1', off.has(k) ? 'border-slate-200 text-slate-400' : 'border-slate-300 text-slate-700')}
            >
              <span className="inline-block h-2.5 w-2.5 rounded-sm border" style={{ background: STYLE[k].fill === 'none' ? 'white' : STYLE[k].fill, borderColor: STYLE[k].stroke }} />
              {STYLE[k].label} <span className="text-slate-400">{plan.layers[k].length}</span>
            </button>
          ))}
          <span className="inline-flex items-center gap-1.5 px-2 py-1 text-slate-700">
            <span className="inline-block h-2.5 w-2.5 rounded-sm border border-rose-700 bg-rose-500/50" /> used in the answer
          </span>
        </div>
      </CardBody>
    </Card>
  );
}
