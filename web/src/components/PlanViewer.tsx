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
};
const ORDER = ['rooms', 'apartments', 'walls', 'windows', 'doors', 'fixtures'];

/** Draws the parsed plan and marks the elements the answer was computed from. */
export function PlanViewer({ drawingId, highlight, hover }: { drawingId: string; highlight: Set<number>; hover: number | null }) {
  const [plan, setPlan] = useState<Plan | null | undefined>(undefined);
  const [off, setOff] = useState<Set<string>>(new Set());

  useEffect(() => {
    setPlan(undefined);
    getPlan(drawingId).then(setPlan).catch(() => setPlan(null));
  }, [drawingId]);

  const layers = useMemo(() => ORDER.filter(k => plan?.layers[k]?.length), [plan]);
  if (plan === undefined) return <Card><CardBody className="text-sm text-slate-500">Loading plan…</CardBody></Card>;
  if (!plan) return null;

  const [, , w, h] = plan.viewBox;
  const toggle = (k: string) => setOff(s => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Plan</CardTitle>
        <span className="text-xs text-slate-500">{highlight.size ? `${highlight.size} highlighted` : 'ask a question to highlight objects'}</span>
      </CardHeader>
      <CardBody className="space-y-3">
        <svg viewBox={`0 0 ${w} ${h}`} className="max-h-[32rem] w-full rounded-md border border-slate-200 bg-white" role="img" aria-label="Floor plan with highlighted objects">
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
