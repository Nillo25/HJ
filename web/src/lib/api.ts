export type Drawing = {
  id: number;
  name: string;
  source: string;
  status: string;
  elements: number;
  capabilities?: { counts?: string; areas?: string; text?: string; reason?: string } | null;
  files?: { format: string; role?: string }[];
};

export type Evidence = {
  kind: string;
  id?: number;
  ref?: string | null;
  type?: string;
  subtype?: string | null;
  label?: string | null;
  floor?: string | null;
  page?: number | null;
  area?: number | null;
  length?: number | null;
  volume?: number | null;
};

export type UsageItem = { model: string; input_tokens: number; output_tokens: number; cost_usd: number };

export type Answer = {
  status: 'answered' | 'estimated' | 'refused' | 'needs_clarification';
  answer?: number | string;
  unit?: string;
  confidence?: number;
  route?: string;
  evidence?: Evidence[];
  usage?: UsageItem[];
  cost_usd?: number;
  latency_ms?: number;
  fn?: string;
  args?: Record<string, unknown>;
  reason?: string;
  missing?: string;
  searched?: string;
};

export type Plan = {
  viewBox: [number, number, number, number];
  layers: Record<string, { id: number; label?: string | null; polygon?: number[][]; bbox?: number[] }[]>;
};

export type UsageTotals = {
  questions: number;
  template_hits: number;
  models: (UsageItem & { calls: number })[];
};

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

export const getDrawings = () => fetch('/api/drawings?limit=500').then(r => j<{ drawings: Drawing[] }>(r)).then(d => d.drawings);
export const getDrawing = (id: number) => fetch(`/api/drawings/${id}`).then(r => j<Drawing>(r));
export const getPlan = (id: number) => fetch(`/api/drawings/${id}/plan`).then(r => (r.ok ? (r.json() as Promise<Plan>) : null));
export const getUsage = () => fetch('/api/usage').then(r => j<UsageTotals>(r));

export const ask = (drawing: string, question: string) =>
  fetch('/api/ask', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ drawing, question }),
  }).then(r => j<Answer>(r));

export async function upload(files: File[]): Promise<string> {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  const r = await j<{ jobId: string }>(await fetch('/api/upload', { method: 'POST', body: fd }));
  return r.jobId;
}

export type Job = { stage: string; log: string[]; finished: boolean; drawings: Drawing[] };

/** Server-sent progress; resolves with the final job state. */
export function watchJob(id: string, onUpdate: (j: Job) => void): Promise<Job> {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/jobs/${id}`);
    es.onmessage = e => onUpdate(JSON.parse(e.data));
    es.addEventListener('done', e => {
      es.close();
      resolve(JSON.parse((e as MessageEvent).data));
    });
    es.onerror = () => {
      es.close();
      reject(new Error('lost connection to the job stream'));
    };
  });
}

export const unitLabel = (u?: string) => (u === 'm2' ? 'm²' : u === 'sf' ? 'SF' : u === 'count' || u === 'text' || !u ? '' : u);
