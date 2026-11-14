/**
 * Raster plans (PNG, and PDF scans the browser turns into PNG) are read by the floor-plan service in raster/
 * (YOLO + a model for the sheet, FastAPI). This file is the only place that knows its API: it uploads the image,
 * asks questions and turns its scene and answers into ours (Answer, Evidence, Plan).
 *
 * RASTER_URL empty = no service: a PNG is shown as uploaded and questions are refused with a reason, as before.
 * The service has no authentication and no limits of its own: keep it on a private network, our limits stand in front.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { SOURCE_BY_KIND, metaPath, type Session } from './db.js';
import { Semaphore, limit } from './limits.js';
import { assertBudget, recordUsage, type Usage } from './usage.js';

export const RASTER_URL = (process.env.RASTER_URL ?? '').replace(/\/+$/, '');
export const rasterEnabled = () => RASTER_URL !== '';
const ANALYZE_TIMEOUT_MS = Number(process.env.RASTER_TIMEOUT_S ?? 180) * 1000;
const ASK_TIMEOUT_MS = 90_000;
/** Reading a sheet is heavy (a model and several calls): few at once, the rest wait. */
export const rasterSlots = new Semaphore(limit('RASTER_PARALLEL', 2));

export class RasterError extends Error {
  constructor(message: string, readonly status = 502) { super(message); }
}

const ID_RE = /^[\w-]+$/;
const MODEL_NAME = 'floorplan-service';

/* ------------------------------------------------------------------ service API */

async function call(pathname: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<Response> {
  if (!rasterEnabled()) throw new RasterError('The floor-plan service is not configured (RASTER_URL).', 503);
  const { timeoutMs = ASK_TIMEOUT_MS, ...rest } = init;
  let r: Response;
  try {
    r = await fetch(`${RASTER_URL}${pathname}`, { ...rest, signal: AbortSignal.timeout(timeoutMs) });
  } catch (e: any) {
    const timeout = e?.name === 'TimeoutError' || e?.name === 'AbortError';
    throw new RasterError(timeout ? `The floor-plan service did not answer in ${Math.round(timeoutMs / 1000)} s.` : 'The floor-plan service is unavailable.');
  }
  if (!r.ok) {
    let why = `${r.status}`;
    try { const j: any = await r.json(); why = typeof j?.detail === 'string' ? j.detail : JSON.stringify(j?.detail ?? j); } catch { /* not json */ }
    throw new RasterError(`Floor-plan service: ${why}`, r.status === 404 ? 404 : 502);
  }
  return r;
}

const jsonPost = (body: unknown): RequestInit => ({
  method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
});

/** Upload the image; the service answers with the scene id (name + hash of the file: the same file gives the same id). */
async function upload(file: string, name: string): Promise<string> {
  const fd = new FormData();
  fd.append('file', new Blob([readFileSync(file)]), name);
  const r = await call('/api/plans', { method: 'POST', body: fd, timeoutMs: ANALYZE_TIMEOUT_MS });
  const id = (await r.json() as { id?: string }).id;
  if (!id || !ID_RE.test(id)) throw new RasterError('Floor-plan service returned no plan id.');
  return id;
}

function patchMeta(s: Session, patch: Record<string, unknown>) {
  Object.assign(s.meta, patch);
  writeFileSync(metaPath(s.uid), JSON.stringify(s.meta));
}

const sourceFile = (s: Session) => path.join(s.dir, 'data', SOURCE_BY_KIND.png, s.meta.fileName);

/** Read the sheet (once per upload; an existing scene is reused by the service). Budget and parallelism are enforced here. */
export async function analyze(s: Session): Promise<string> {
  if (s.meta.raster && ID_RE.test(s.meta.raster)) return s.meta.raster;
  assertBudget();
  const file = sourceFile(s);
  if (!existsSync(file)) throw new RasterError('The uploaded image is gone.', 404);
  await rasterSlots.acquire();
  try {
    const id = await upload(file, s.meta.fileName);
    patchMeta(s, { raster: id });
    return id;
  } finally { rasterSlots.release(); }
}

/** The scene was lost on the service side (restart): read the sheet again from our copy of the image. */
async function reanalyzeFromFile(s: Session): Promise<string> {
  delete s.meta.raster; scenes.delete(s.uid);
  return analyze(s);
}

/* ------------------------------------------------------------------ scene -> our shapes */

type RObject = { id: string; class: string; polygon: number[][]; conf: number; room?: string | null };
export type RScene = { width: number; height: number; meta?: Record<string, any>; objects: RObject[]; rooms: RObject[]; stale?: boolean };

const scenes = new Map<string, { at: number; rid: string; scene: RScene }>();
const SCENE_TTL_MS = 30_000;

async function getScene(s: Session, rid: string): Promise<RScene> {
  const hit = scenes.get(s.uid);
  if (hit && hit.rid === rid && Date.now() - hit.at < SCENE_TTL_MS) return hit.scene;
  const scene = await (await call(`/api/plans/${rid}`)).json() as RScene;
  scenes.set(s.uid, { at: Date.now(), rid, scene });
  return scene;
}

/** Their classes -> layers of PlanViewer. */
const LAYER: Record<string, string> = {
  wall: 'walls', door: 'doors', window: 'windows',
  toilet: 'fixtures', sink: 'fixtures', bathtub: 'fixtures', shower: 'fixtures', sauna_bench: 'fixtures',
};
const layerOf = (cls: string) => (cls.startsWith('room_') ? 'rooms' : LAYER[cls] ?? 'fixtures');

const bboxOf = (poly: number[][]): number[] | null => {
  if (!poly?.length) return null;
  const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)].map((v) => Math.round(v * 10) / 10);
};

/** Numeric ids (ours): objects first, then rooms, in the order of the scene. */
function numbering(scene: RScene): Map<string, number> {
  const m = new Map<string, number>();
  for (const o of [...scene.objects, ...scene.rooms]) if (!m.has(o.id)) m.set(o.id, m.size + 1);
  return m;
}

export function planFromScene(scene: RScene, background: string) {
  const ids = numbering(scene);
  const layers: Record<string, { id: number; label: string; polygon: number[][]; bbox: number[] }[]> = {};
  for (const o of [...scene.rooms, ...scene.objects]) {
    const bb = bboxOf(o.polygon);
    if (!bb) continue;
    (layers[layerOf(o.class)] ??= []).push({ id: ids.get(o.id)!, label: `${o.class.replace('room_', '')} ${o.id}`, polygon: o.polygon, bbox: bb });
  }
  return { viewBox: [0, 0, scene.width, scene.height] as number[], background, overlay: true, layers };
}

function evidenceOf(scene: RScene, found: string[]) {
  const ids = numbering(scene);
  const byId = new Map([...scene.objects, ...scene.rooms].map((o) => [o.id, o]));
  const seen = new Set<string>();
  const out: any[] = [];
  for (const rid of found) {
    const o = byId.get(rid);
    if (!o || seen.has(rid)) continue;
    seen.add(rid);
    const isRoom = o.class.startsWith('room_');
    const bb = bboxOf(o.polygon);
    out.push({
      kind: 'raster', id: ids.get(rid), ref: rid,
      type: isRoom ? 'room' : o.class, subtype: isRoom ? o.class.replace('room_', '') : null,
      label: !isRoom && o.room ? `in ${o.room}` : null,
      bbox: bb ? JSON.stringify(bb) : null, page: null, area: null, length: null, volume: null,
      confidence: Math.round(o.conf * 100) / 100,
    });
  }
  return out;
}

/* ------------------------------------------------------------------ answers */

type RChat = {
  reply: string; action?: { type?: string; ids?: string[] }; confidence?: 'high' | 'medium' | 'low' | null;
  calls?: { tool: string; args?: Record<string, unknown>; result?: { answer?: unknown; unit?: string | null; ids?: string[]; note?: string | null; evidence?: { id?: string }[] } }[];
  parsed_by?: string; answered_by?: string | null; cost_usd?: number; latency_ms?: number;
};

const CONFIDENCE = { high: 0.95, medium: 0.8, low: 0.5 } as const;
/** Their units (Russian "шт", "px2" when no scale is set) -> ours. */
const UNIT: Record<string, string> = { 'шт': 'count', 'шт.': 'count', px2: 'px²', 'м2': 'm2', 'м²': 'm2', 'м': 'm' };
const unitOf = (u?: string | null) => (u ? UNIT[u.toLowerCase()] ?? u : 'count');

/** Their chat reply -> our Answer. Pure: tested without the service. */
export function toAnswer(chat: RChat, scene: RScene, usage: Usage[] = []) {
  const calls = chat.calls ?? [];
  const found = [...new Set([
    ...(chat.action?.ids ?? []),
    ...calls.flatMap((c) => c.result?.ids ?? []),
    ...calls.flatMap((c) => (c.result?.evidence ?? []).map((e) => e.id).filter((x): x is string => !!x)),
  ])];
  const evidence = evidenceOf(scene, found);
  const cost = chat.cost_usd ?? 0;
  const base = {
    route: cost > 0 ? 'llm' : 'template', engine: 'raster', usage, cost_usd: cost, latency_ms: chat.latency_ms,
    fn: calls[0]?.tool, args: calls[0]?.args,
  };
  const noValue = (a: unknown) => a == null || (Array.isArray(a) && a.length === 0);
  if (!calls.length || chat.parsed_by === 'none' || (!found.length && calls.every((c) => noValue(c.result?.answer)))) {
    return { ...base, status: 'refused' as const, reason: chat.reply, missing: 'question', evidence: [] };
  }
  const withAnswer = [...calls].reverse().find((c) => c.result?.answer != null);
  const raw = withAnswer?.result?.answer;
  const numeric = typeof raw === 'number' ? raw : typeof raw === 'string' && raw.trim() !== '' && !isNaN(Number(raw)) ? Number(raw) : null;
  const level = chat.confidence ?? 'low';
  const note = [...calls.map((c) => c.result?.note).filter(Boolean), level !== 'high' ? 'Read from a raster image: check on the plan.' : null].join(' ') || undefined;
  return {
    ...base,
    status: (level === 'high' ? 'answered' : 'estimated') as 'answered' | 'estimated',
    answer: numeric ?? chat.reply, unit: numeric != null ? unitOf(withAnswer?.result?.unit) : undefined,
    confidence: CONFIDENCE[level], evidence, note,
  };
}

export async function rasterAsk(s: Session, question: string, t0: number) {
  if (!rasterEnabled()) {
    return { status: 'refused', reason: 'raster plans are not read in this deployment (no floor-plan service configured)',
             missing: 'source_model', route: 'none', usage: [], cost_usd: 0, latency_ms: Date.now() - t0 };
  }
  let rid = await analyze(s);
  let r: Response;
  try {
    r = await call(`/api/plans/${rid}/chat`, jsonPost({ message: question, history: [] }));
  } catch (e) {
    if (!(e instanceof RasterError) || e.status !== 404) throw e;
    rid = await reanalyzeFromFile(s);                    // the service forgot the scene: read the sheet again, once
    r = await call(`/api/plans/${rid}/chat`, jsonPost({ message: question, history: [] }));
  }
  const chat = await r.json() as RChat;
  const used = (chat.cost_usd ?? 0) > 0 ? [recordUsage(MODEL_NAME, 0, 0, chat.cost_usd!)] : [];
  const scene = await getScene(s, rid);
  const out = toAnswer(chat, scene, used);
  return { ...out, latency_ms: Date.now() - t0 };
}

/* ------------------------------------------------------------------ viewer data and scale */

export async function rasterPlan(s: Session) {
  const rid = await analyze(s);
  return planFromScene(await getScene(s, rid), `/api/drawings/${s.uid}/image`);
}

export async function rasterImage(s: Session): Promise<Response> {
  const rid = await analyze(s);
  return call(`/api/plans/${rid}/image`);
}

/** What the screen needs to say about the reading: scale, whether it is out of date. */
export async function rasterInfo(s: Session) {
  const rid = await analyze(s);
  const sc = await getScene(s, rid);
  const scale = sc.meta?.scale ?? {};
  return {
    px_per_m: scale.px_per_m ?? null, scale_source: scale.source ?? null, scale_detail: scale.detail ?? null,
    stale: Boolean(sc.stale), size: [sc.width, sc.height], objects: sc.objects.length, rooms: sc.rooms.length,
  };
}

export type ScaleIn = { x1: number; y1: number; x2: number; y2: number; length: number; unit: string };

export async function rasterSetScale(s: Session, b: ScaleIn) {
  const rid = await analyze(s);
  const r = await (await call(`/api/plans/${rid}/scale`, jsonPost(b))).json();
  scenes.delete(s.uid);
  return r;
}

export async function rasterAutoScale(s: Session) {
  assertBudget();
  const rid = await analyze(s);
  const r = await (await call(`/api/plans/${rid}/scale/auto`, { method: 'POST', timeoutMs: ANALYZE_TIMEOUT_MS })).json();
  scenes.delete(s.uid);
  return r;
}

export async function rasterReanalyze(s: Session) {
  assertBudget();
  const rid = await analyze(s);
  await rasterSlots.acquire();
  try { await call(`/api/plans/${rid}/reanalyze`, { method: 'POST', timeoutMs: ANALYZE_TIMEOUT_MS }); }
  finally { rasterSlots.release(); }
  scenes.delete(s.uid);
  return { ok: true };
}
