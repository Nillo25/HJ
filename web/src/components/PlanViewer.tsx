import { useEffect, useMemo, useState } from 'react';
import { getPlan, type Evidence, type Plan } from '../lib/api';
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
const FOUND = '#e11d48';
const MAX_BADGES = 99;

type Item = { id: number; label?: string | null; polygon?: number[][]; bbox?: number[] };
const centre = (it: Item): [number, number] | null => {
  if (it.bbox) return [(it.bbox[0] + it.bbox[2]) / 2, (it.bbox[1] + it.bbox[3]) / 2];
  if (it.polygon?.length) {
    const n = it.polygon.length;
    return [it.polygon.reduce((s, p) => s + p[0], 0) / n, it.polygon.reduce((s, p) => s + p[1], 0) / n];
  }
  return null;
};
// "toilet" and "kitchen" say more than "fixture" and "room"; "single" says nothing about a door
const nameOf = (e: Evidence) => {
  const t = (e.type ?? '').replace('_row', ' row') || 'object';
  return (t === 'fixture' || t === 'room') && e.subtype ? e.subtype : t;
};

/**
 * Draws the plan and proves the answer on it: what was found is marked in red and numbered like the
 * evidence table, everything else is dimmed, and per storey counts say where the objects are.
 */
export function PlanViewer({ drawingId, evidence }: { drawingId: string; evidence: Evidence[] }) {
  const [plan, setPlan] = useState<Plan | null | undefined>(undefined);
  const [floor, setFloor] = useState<number | null>(null);
  const [off, setOff] = useState<Set<string>>(new Set());
  const [fit, setFit] = useState(true);

  // number of each found object = its row in the evidence table
  const numbers = useMemo(() => {
    const m = new Map<number, number>();
    evidence.forEach((e, i) => { if (e.id != null && !m.has(e.id)) m.set(e.id, i + 1); });
    return m;
  }, [evidence]);

  const byFloor = useMemo(() => {
    const m = new Map<number, number>();
    for (const e of evidence) if (e.floor_id != null) m.set(e.floor_id, (m.get(e.floor_id) ?? 0) + 1);
    return m;
  }, [evidence]);
  const kinds = useMemo(() => {
    const c = new Map<string, number>();
    for (const e of evidence) c.set(nameOf(e), (c.get(nameOf(e)) ?? 0) + 1);
    return [...c.entries()].map(([k, n]) => `${k} × ${n}`).join(', ');
  }, [evidence]);
  const undrawn = useMemo(() => evidence.filter(e => !e.bbox).length, [evidence]);

  useEffect(() => { setFloor(null); }, [drawingId]);
  // follow the answer to the storey that has most of the found objects
  useEffect(() => {
    if (!byFloor.size) return;
    setFloor([...byFloor.entries()].reduce((a, b) => (b[1] > a[1] ? b : a))[0]);
    setFit(true);
  }, [byFloor]);
  useEffect(() => { getPlan(drawingId, floor).then(setPlan).catch(() => setPlan(null)); }, [drawingId, floor]);

  const layers = useMemo(() => ORDER.filter(k => plan?.layers[k]?.length), [plan]);
  const found = useMemo(() => {
    const out: { key: string; it: Item; n: number }[] = [];
    if (plan) for (const k of Object.keys(plan.layers)) for (const it of plan.layers[k] as Item[]) {
      const n = numbers.get(it.id);
      if (n != null) out.push({ key: k, it, n });
    }
    return out.sort((a, b) => a.n - b.n);
  }, [plan, numbers]);

  const focus = useMemo(() => {
    if (!found.length) return null;
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const { it } of found) {
      const pts = it.polygon ?? (it.bbox ? [[it.bbox[0], it.bbox[1]], [it.bbox[2], it.bbox[3]]] : []);
      for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
    }
    if (!isFinite(x0)) return null;
    const pad = Math.max(x1 - x0, y1 - y0) * 0.08 + 10;
    return [x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad];
  }, [found]);

  if (plan === undefined) return <Card><CardBody className="text-sm text-slate-500">Loading plan…</CardBody></Card>;
  if (!plan) return null;

  const [, , w, h] = plan.viewBox;
  const vb = focus && fit ? focus : [0, 0, w, h];
  const r = Math.max(vb[2], vb[3]) * 0.015;            // badge radius follows the zoom
  const toggle = (k: string) => setOff(s => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });
  const anyFound = numbers.size > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{anyFound ? `Found: ${kinds}` : 'Plan'}</CardTitle>
        <span className="flex items-center gap-2 text-xs text-slate-500">
          {anyFound && `${found.length} shown here`}
          {focus && (
            <button onClick={() => setFit(f => !f)} className="rounded-md border border-slate-300 px-2 py-0.5 text-slate-700 hover:bg-slate-100">
              {fit ? 'Show whole plan' : 'Zoom to answer'}
            </button>
          )}
        </span>
      </CardHeader>
      <CardBody className="space-y-3">
        {plan.floors && plan.floors.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-slate-500">Storey</span>
            {plan.floors.map(f => {
              const n = byFloor.get(f.id) ?? 0;
              const active = plan.floor === f.id;
              return (
                <button key={f.id} onClick={() => { setFloor(f.id); setFit(true); }} aria-pressed={active}
                  className={cn('rounded-md border px-2 py-1', active ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-300 text-slate-700 hover:bg-slate-100')}>
                  {f.name || `#${f.id}`}
                  {anyFound && <span className={cn('ml-1.5 rounded px-1', n ? (active ? 'bg-rose-400 text-white' : 'bg-rose-100 text-rose-700') : 'text-slate-400')}>{n}</span>}
                </button>
              );
            })}
            {plan.coverage && (
              <span className="ml-1 text-slate-500" title="Objects whose shape could be read from the file. The rest are counted but not drawn.">
                drawn {plan.coverage.drawn} of {plan.coverage.total}
              </span>
            )}
          </div>
        )}
        {undrawn > 0 && (
          <p className="rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800">
            {undrawn} of {evidence.length} found objects have no readable shape in the file: they are counted and listed in the table, but not marked on the plan.
          </p>
        )}
        <svg viewBox={vb.join(' ')} className="max-h-[32rem] w-full rounded-md border border-slate-200 bg-white" role="img" aria-label="Plan with the found objects marked">
          {plan.background && <image href={plan.background} x={0} y={0} width={w} height={h} opacity={anyFound ? 0.55 : 1} />}
          {layers.filter(k => !off.has(k)).map(k => (
            <g key={k} opacity={anyFound ? 0.4 : 1}>
              {(plan.layers[k] as Item[]).filter(it => !numbers.has(it.id)).map(it => shape(it, k, STYLE[k].fill, STYLE[k].stroke, STYLE[k].width))}
            </g>
          ))}
          {/* found objects on top, full strength */}
          <g>
            {found.map(({ key, it }) => shape(it, key, 'rgba(225,29,72,0.55)', FOUND, Math.max(STYLE[key].width, 1) + 1.5))}
          </g>
          {found.length <= MAX_BADGES && key_badges(found, r)}
        </svg>
        <div className="flex flex-wrap gap-2 text-xs">
          {layers.map(k => (
            <button key={k} onClick={() => toggle(k)} aria-pressed={!off.has(k)}
              className={cn('inline-flex items-center gap-1.5 rounded-md border px-2 py-1', off.has(k) ? 'border-slate-200 text-slate-400' : 'border-slate-300 text-slate-700')}>
              <span className="inline-block h-2.5 w-2.5 rounded-sm border" style={{ background: STYLE[k].fill === 'none' ? 'white' : STYLE[k].fill, borderColor: STYLE[k].stroke }} />
              {STYLE[k].label} <span className="text-slate-400">{plan.layers[k].length}</span>
            </button>
          ))}
          {anyFound && (
            <span className="inline-flex items-center gap-1.5 px-2 py-1 font-medium text-rose-700">
              <span className="inline-block h-2.5 w-2.5 rounded-sm border border-rose-700 bg-rose-500/60" /> found, numbered as in the table
            </span>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

function shape(it: Item, k: string, fill: string, stroke: string, width: number) {
  const common = { fill, stroke, strokeWidth: width, vectorEffect: 'non-scaling-stroke' as const };
  if (it.polygon?.length) return <polygon key={it.id} points={it.polygon.map(p => p.join(',')).join(' ')} {...common}><title>{it.label ?? k}</title></polygon>;
  if (it.bbox) return <rect key={it.id} x={it.bbox[0]} y={it.bbox[1]} width={it.bbox[2] - it.bbox[0]} height={it.bbox[3] - it.bbox[1]} {...common}><title>{it.label ?? k}</title></rect>;
  return null;
}

function key_badges(found: { it: Item; n: number }[], r: number) {
  return (
    <g fontFamily="system-ui, sans-serif" textAnchor="middle" pointerEvents="none">
      {found.map(({ it, n }) => {
        const c = centre(it);
        if (!c) return null;
        return (
          <g key={`b${it.id}`}>
            <circle cx={c[0]} cy={c[1]} r={r} fill={FOUND} stroke="#fff" strokeWidth={r * 0.18} />
            <text x={c[0]} y={c[1] + r * 0.36} fontSize={r * (n > 9 ? 0.95 : 1.15)} fontWeight={700} fill="#fff">{n}</text>
          </g>
        );
      })}
    </g>
  );
}
