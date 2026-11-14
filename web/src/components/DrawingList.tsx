import type { Drawing } from '../lib/api';
import { Badge, Card, CardBody, CardHeader, CardTitle, cn } from './ui';

const LABEL: Record<string, string> = {
  swiss_dwellings: 'DXF plan',
  ifc_models: 'IFC model',
  ifc_bench: 'IFC report',
  opening_design: 'PDF sheet',
  cubicasa5k: 'PNG plan',
};

export function DrawingList({ drawings, selected, onSelect }: { drawings: Drawing[]; selected: number | null; onSelect: (d: Drawing) => void }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Drawings</CardTitle>
        <span className="text-xs text-slate-500">{drawings.length}</span>
      </CardHeader>
      <CardBody className="max-h-80 space-y-1 overflow-y-auto p-2">
        {!drawings.length && <p className="p-2 text-sm text-slate-500">Nothing indexed yet. Upload a file.</p>}
        {drawings.map(d => (
          <button
            key={d.id}
            onClick={() => onSelect(d)}
            aria-pressed={selected === d.id}
            className={cn(
              'flex w-full flex-col items-start gap-1 rounded-md px-2 py-1.5 text-left transition-colors',
              selected === d.id ? 'bg-slate-900 text-white' : 'hover:bg-slate-100',
            )}
          >
            <span className="w-full truncate text-sm font-medium" title={d.name}>{d.name}</span>
            <span className="flex items-center gap-2">
              <Badge tone={selected === d.id ? 'blue' : 'slate'}>{LABEL[d.source] ?? 'upload'}</Badge>
              <span className={cn('text-xs', selected === d.id ? 'text-slate-300' : 'text-slate-500')}>
                {d.elements ? `${d.elements} objects` : 'not parsed'}
              </span>
            </span>
          </button>
        ))}
      </CardBody>
    </Card>
  );
}
