import { useEffect, useState } from 'react';
import { getUsage, type UsageTotals } from '../lib/api';
import { Badge } from './ui';

/** Inference is judged, so the cost of every model is always on screen. */
export function UsageBar({ refreshKey }: { refreshKey: number }) {
  const [u, setU] = useState<UsageTotals | null>(null);
  useEffect(() => {
    getUsage().then(setU).catch(() => setU(null));
  }, [refreshKey]);
  if (!u) return null;

  const tokens = u.models.reduce((n, m) => n + m.input_tokens + m.output_tokens, 0);
  const usd = u.models.reduce((n, m) => n + m.cost_usd, 0);
  const share = u.questions ? Math.round((u.template_hits / u.questions) * 100) : null;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600" aria-label="Inference usage this session">
      <span><b className="tabular-nums">{u.questions}</b> question{u.questions === 1 ? '' : 's'}</span>
      {share != null && <span><b className="tabular-nums">{share}%</b> without a model</span>}
      <span><b className="tabular-nums">{tokens.toLocaleString('en-US')}</b> tokens</span>
      <span><b className="tabular-nums">${usd.toFixed(4)}</b></span>
      {u.models.map(m => (
        <Badge key={m.model} tone="blue">
          {m.model} · {m.calls} call{m.calls === 1 ? '' : 's'} · {m.input_tokens.toLocaleString('en-US')} in / {m.output_tokens.toLocaleString('en-US')} out
        </Badge>
      ))}
      {!u.models.length && <Badge tone="green">no model calls yet</Badge>}
    </div>
  );
}
