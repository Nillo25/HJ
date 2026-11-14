import { unitLabel, type Answer, type Evidence } from '../lib/api';
import { Card, CardHeader, CardTitle, Table, Td, Th, cn } from './ui';

const TYPE: Record<string, string> = { room_row: 'schedule room', door_row: 'schedule door' };
const value = (e: Evidence) => e.area ?? e.length ?? e.volume ?? null;

export function EvidenceTable({ answer, hover, onHover }: { answer: Answer; hover: number | null; onHover: (id: number | null) => void }) {
  const rows = answer.evidence ?? [];
  if (!rows.length) return null;

  const hasValue = rows.some(r => value(r) != null);
  const sum = rows.reduce((s, r) => s + (value(r) ?? 0), 0);
  const hasFloor = rows.some(r => r.floor);
  const total = typeof answer.answer === 'number' ? answer.answer : null;
  // the table is capped server-side; say so when the answer counts more rows than are listed
  const capped = answer.unit === 'count' && total != null && total > rows.length;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Evidence</CardTitle>
        <span className="text-xs text-slate-500">
          {capped ? `first ${rows.length} of ${total}` : `${rows.length} row${rows.length === 1 ? '' : 's'}`}
        </span>
      </CardHeader>
      <div className="max-h-96 overflow-auto">
        <Table>
          <thead>
            <tr>
              <Th className="w-10">#</Th>
              <Th>Type</Th>
              <Th>Label</Th>
              {hasFloor && <Th>Floor</Th>}
              {hasValue && <Th className="text-right">{answer.unit === 'm2' ? 'Area, m²' : 'Value'}</Th>}
              <Th>Ref</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr
                key={`${r.id ?? r.ref ?? i}-${i}`}
                onMouseEnter={() => r.id != null && onHover(r.id)}
                onMouseLeave={() => onHover(null)}
                className={cn(hover != null && hover === r.id && 'bg-sky-50')}
              >
                <Td className="tabular-nums text-slate-400">{i + 1}</Td>
                <Td>{[TYPE[r.type ?? ''] ?? r.type, r.subtype].filter(Boolean).join(' · ') || r.kind}</Td>
                <Td className="max-w-56 truncate" title={r.label ?? ''}>{r.label ?? '—'}</Td>
                {hasFloor && <Td>{r.floor ?? '—'}</Td>}
                {hasValue && <Td className="text-right tabular-nums">{value(r) != null ? Number(value(r)).toFixed(2) : '—'}</Td>}
                <Td className="max-w-32 truncate font-mono text-xs text-slate-500" title={r.ref ?? ''}>{r.ref ?? (r.id != null ? `#${r.id}` : '—')}</Td>
              </tr>
            ))}
          </tbody>
          {hasValue && !capped && (
            <tfoot>
              <tr className="bg-slate-50 font-medium">
                <Td colSpan={hasFloor ? 4 : 3}>Total</Td>
                <Td className="text-right tabular-nums">{sum.toFixed(2)}</Td>
                <Td>{unitLabel(answer.unit)}</Td>
              </tr>
            </tfoot>
          )}
        </Table>
      </div>
    </Card>
  );
}
