export type Kind = 'dxf' | 'ifc' | 'pdf' | 'png';

export type Drawing = {
  id: string;
  name: string;
  kind: Kind;
  source: string;
  status: string;
  elements: number;
  text_lines?: number;
  vector?: boolean;
  /** a raster plan (PNG) read by the floor-plan service: it is drawn from the scene it returned */
  raster?: boolean;
  /** what reading the sheet cost in Claude calls (dollars); the service reports money, not tokens */
  raster_cost_usd?: number;
  /** when the file was uploaded (ms since epoch); uploads are kept on the server for UPLOAD_TTL_HOURS */
  created_at?: number;
  capabilities?: { counts?: string; areas?: string; text?: string; reason?: string } | null;
  files?: { format: string; role?: string }[];
};

export type Evidence = {
  kind: string;
  id?: number;
  floor_id?: number | null;
  bbox?: string | null;
  ref?: string | null;
  type?: string;
  subtype?: string | null;
  label?: string | null;
  floor?: string | null;
  page?: number | null;
  area?: number | null;
  length?: number | null;
  volume?: number | null;
  /** detection confidence of this object (0..1): raster plans only; vector data has none */
  confidence?: number | null;
  /** raster: a borderline detection (under its class threshold + margin), drawn yellow */
  weak?: boolean;
};

type UsageItem = { model: string; input_tokens: number; output_tokens: number; cost_usd: number };

export type Part = { group: string; label: string; count: number; area: number; evidence: Evidence[] };
export type Alternative = { label: string; answer: number; unit?: string; confidence?: number; evidence: Evidence[] };

export type Answer = {
  question?: string;
  alternatives?: Alternative[];
  breakdown?: Part[];
  note?: string;
  status: 'answered' | 'estimated' | 'refused' | 'needs_clarification';
  answer?: number | string;
  unit?: string;
  confidence?: number;
  route?: string;
  /** 'raster': read by the floor-plan service (Claude, reported in dollars only) */
  engine?: string;
  /** raster: the question corrected the plan (an object removed, a room retyped): draw the plan again */
  scene_changed?: boolean;
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
  background?: string;
  /** with a background: draw the layers over it (detections over a raster plan) */
  overlay?: boolean;
  floors?: { id: number; name: string; n: number }[];
  floor?: number | null;
  coverage?: { drawn: number; total: number };
  layers: Record<string, { id: number; label?: string | null; polygon?: number[][]; bbox?: number[] }[]>;
};

export type UsageTotals = {
  questions: number;
  template_hits: number;
  models: (UsageItem & { calls: number; failed_calls?: number })[];
};

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

/** Every file uploaded to this server and still kept, newest first (the server sorts). */
export const listDrawings = () => fetch('/api/drawings').then(r => j<{ drawings: Drawing[] }>(r)).then(b => b.drawings);
export const getDrawing = (id: string) => fetch(`/api/drawings/${id}`).then(r => j<Drawing>(r));
export const getPlan = (id: string, floor?: number | null) => fetch(`/api/drawings/${id}/plan${floor != null ? `?floor=${floor}` : ''}`).then(r => (r.ok ? (r.json() as Promise<Plan>) : null));
export const getUsage = () => fetch('/api/usage').then(r => j<UsageTotals>(r));

export const ask = (drawingId: string, question: string) =>
  fetch('/api/ask', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ drawing_id: drawingId, question }),
  }).then(r => j<Answer>(r));

/** Upload with progress (fetch cannot report it): resolves with the job id. */
export function upload(kind: Kind, file: File, onProgress?: (fraction: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open('POST', '/api/upload');
    x.upload.onprogress = e => { if (e.lengthComputable) onProgress?.(e.loaded / e.total); };
    x.onload = () => {
      let body: { jobId?: string; error?: string } = {};
      try { body = JSON.parse(x.responseText); } catch { /* not JSON */ }
      if (x.status >= 200 && x.status < 300 && body.jobId) resolve(body.jobId);
      else reject(new Error(body.error ?? `${x.status} ${x.statusText}`));
    };
    x.onerror = () => reject(new Error('Network error: the server did not answer.'));
    const fd = new FormData();
    fd.append('kind', kind);
    fd.append('files', file);
    x.send(fd);
  });
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

export type RasterInfo = {
  px_per_m: number | null; scale_source: string | null; scale_detail: string | null;
  stale: boolean; size: [number, number]; objects: number; rooms: number;
};
async function post<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, { method: 'POST', headers: body ? { 'content-type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) throw new Error(((await r.json().catch(() => null)) as { error?: string } | null)?.error ?? `${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}
export type ClearResult = { files: number; templates: number; kept_busy: number; raster: { plans: number } | { error: string } | null };
/** Delete every upload, the question cache and the floor-plan service's scenes. */
export const clearCache = () => post<ClearResult>('/api/cache/clear');
export const getRaster = (id: string) => fetch(`/api/drawings/${id}/raster`).then(r => j<RasterInfo>(r));
export type ScaleBody = { x1: number; y1: number; x2: number; y2: number; length: number; unit: string };
export const setRasterScale = (id: string, b: ScaleBody) => post<{ px_per_m: number }>(`/api/drawings/${id}/raster/scale`, b);
export const autoRasterScale = (id: string) => post<{ px_per_m?: number | null; detail?: string }>(`/api/drawings/${id}/raster/scale/auto`);
export const addRasterObject = (id: string, x: number, y: number, cls: string) =>
  post<{ id: string; cls: string; count: number | null; how: string; cost_usd: number }>(`/api/drawings/${id}/raster/objects`, { x, y, cls });
export const undoRaster = (id: string) => post<{ undone: string | null; note: string; changed: boolean }>(`/api/drawings/${id}/raster/undo`);
export const reanalyzeRaster = (id: string) => post<{ ok: boolean }>(`/api/drawings/${id}/raster/reanalyze`);

export const unitLabel = (u?: string) => (u === 'm2' ? 'm²' : u === 'sf' ? 'SF' : u === 'count' || u === 'text' || !u ? '' : u);
