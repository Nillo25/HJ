import { serve } from '@hono/node-server';
import { serveStatic } from '@hono/node-server/serve-static';
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { streamSSE } from 'hono/streaming';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { Readable } from 'node:stream';
import path from 'node:path';

import {
  KINDS, SOURCE_BY_KIND, UPLOADS, cleanupUploads, elementCount, ensureUploads, filesOf, floorsOf,
  getSession, latestEval, listSessions, queryElements, metaPath,
  type Element, type Kind, type Session,
} from './db.js';
import { ask } from './ask.js';
import { PYTHON } from './py.js';
import { countQuestion, usageSnapshot } from './usage.js';

ensureUploads();
cleanupUploads();
setInterval(cleanupUploads, 3600_000).unref();

const MAX_MB = Number(process.env.UPLOAD_MAX_MB ?? 100);
const INDEX_TIMEOUT_S = Number(process.env.INDEX_TIMEOUT_S ?? 120);
const EXT_BY_KIND: Record<Kind, string> = { dxf: '.dxf', ifc: '.ifc', pdf: '.pdf', png: '.png' };

const app = new Hono();
app.use('/api/*', cors());

/* ------------------------------------------------------------ drawings */

function card(s: Session) {
  return {
    id: s.uid, name: s.meta.fileName.replace(/\.[^.]+$/, ''), kind: s.meta.kind,
    source: s.drawing.source, status: s.drawing.status, capabilities: s.drawing.capabilities,
    elements: elementCount(s.db, s.drawing.id), created_at: s.meta.createdAt,
  };
}

app.get('/api/drawings', (c) => c.json({ drawings: listSessions().map(card) }));

app.get('/api/drawings/:id', (c) => {
  const s = getSession(c.req.param('id'));
  if (!s) return c.json({ error: 'not found' }, 404);
  return c.json({
    ...card(s),
    files: filesOf(s.db, s.drawing.id),
    floors: floorsOf(s.db, s.drawing.id),
    counts: s.db.prepare(
      'SELECT type, subtype, COUNT(*) n FROM elements WHERE drawing_id=? GROUP BY type, subtype',
    ).all(s.drawing.id),
  });
});

/** The original upload, for the raster display (PNG). */
app.get('/api/drawings/:id/image', (c) => {
  const s = getSession(c.req.param('id'));
  if (!s || s.meta.kind !== 'png') return c.json({ error: 'no image' }, 404);
  const abs = path.join(s.dir, 'data', SOURCE_BY_KIND.png, s.meta.fileName);
  if (!existsSync(abs)) return c.json({ error: 'no image' }, 404);
  return c.body(Readable.toWeb(createReadStream(abs)) as ReadableStream, 200, {
    'content-type': 'image/png', 'content-length': String(statSync(abs).size),
  });
});

app.get('/api/drawings/:id/sheet.svg', (c) => {
  const s = getSession(c.req.param('id'));
  const abs = s ? path.join(s.dir, 'sheet.svg') : '';
  if (!s || s.meta.kind !== 'pdf' || !existsSync(abs)) return c.json({ error: 'no sheet' }, 404);
  return c.body(Readable.toWeb(createReadStream(abs)) as ReadableStream, 200, {
    'content-type': 'image/svg+xml', 'content-length': String(statSync(abs).size),
  });
});

/* ------------------------- geometry for the plan viewer (contract section 6) */

app.get('/api/drawings/:id/plan', (c) => {
  const s = getSession(c.req.param('id'));
  if (!s) return c.json({ error: 'not found' }, 404);
  const els = queryElements(s.db, {
    drawing_id: s.drawing.id, type: null, subtype: null, floor: null, literal: null,
  }) as Element[];
  if (!els.some((e) => e.bbox)) return c.json({ error: 'no geometry' }, 404);

  // vector PDF sheet: the rendered page is the background, schedule rows are boxes on it
  if (s.meta.kind === 'pdf') {
    const sheet = path.join(s.dir, 'sheet.svg');
    if (!existsSync(sheet)) return c.json({ error: 'sheet was not rendered' }, 404);
    const head = readFileSync(sheet, 'utf8').slice(0, 300).match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
    if (!head) return c.json({ error: 'bad sheet' }, 500);
    const [W, H] = [Number(head[1]), Number(head[2])];
    const rows = els.filter((e) => e.bbox).map((e) => {
      const b = JSON.parse(e.bbox!);
      return { id: e.id, label: e.label, bbox: [b[0], H - b[3], b[2], H - b[1]] };
    });
    return c.json({ viewBox: [0, 0, W, H], background: `/api/drawings/${s.uid}/sheet.svg`, layers: { rows } });
  }

  // overall extent -> screen viewBox 1200 x N, the Y axis is flipped
  let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const e of els) {
    if (!e.bbox) continue;
    const b = JSON.parse(e.bbox);
    x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]);
    x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]);
  }
  const W = 1200;
  const sc = W / Math.max(x1 - x0, 1e-6);
  const H = Math.round((y1 - y0) * sc);
  const tx = ([x, y]: number[]) => [
    Math.round((x - x0) * sc * 10) / 10,
    Math.round((y1 - y) * sc * 10) / 10,
  ];
  const txBox = (b: number[]) => {
    const a = tx([b[0], b[3]]); const d2 = tx([b[2], b[1]]);
    return [a[0], a[1], d2[0], d2[1]];
  };

  const layers: Record<string, any[]> = {
    walls: [], rooms: [], doors: [], windows: [], fixtures: [], apartments: [],
  };
  const bucket: Record<string, string> = {
    wall: 'walls', room: 'rooms', door: 'doors',
    window: 'windows', fixture: 'fixtures', apartment: 'apartments',
  };

  for (const e of els) {
    const key = bucket[e.type];
    if (!key) continue;
    const item: any = { id: e.id, label: e.label, subtype: e.subtype };
    if (e.polygon) item.polygon = (JSON.parse(e.polygon) as number[][]).map(tx);
    if (e.bbox) item.bbox = txBox(JSON.parse(e.bbox));
    if (e.area != null) item.area = Math.round(e.area * 100) / 100;
    if (e.confidence < 1) item.confidence = e.confidence;
    layers[key].push(item);
  }

  return c.json({ viewBox: [0, 0, W, H], layers });
});

/* ------------------------------------------------------------- question */

app.post('/api/ask', async (c) => {
  const body = await c.req.json<{ question: string; drawing_id?: string }>();
  if (!body?.question) return c.json({ error: 'question required' }, 400);
  const s = body.drawing_id ? getSession(body.drawing_id) : null;
  if (!s) return c.json({ error: 'drawing required' }, 400);

  try {
    const res: any = await ask(s, body.question);
    countQuestion(res.route === 'template' || res.route === 'cache');
    return c.json(res);
  } catch (e: any) {
    return c.json({ status: 'refused', reason: `${e?.name}: ${e?.message}`,
                    missing: 'internal', route: 'llm', cost_usd: 0 }, 200);
  }
});

/* ------------------------------------------------------- usage and eval */

app.get('/api/usage', (c) => c.json(usageSnapshot()));
app.get('/api/eval/latest', (c) => c.json({ rows: latestEval() }));

/* ------------------------------------------------- upload and indexing */

type Job = { id: string; stage: string; log: string[]; finished: boolean; drawings: any[] };
const jobs = new Map<string, Job>();

function finish(job: Job, ok: boolean, why?: string) {
  job.finished = true;
  job.stage = ok ? 'ready' : 'failed';
  if (!ok && why) job.log.push(why);
  if (ok) { const s = getSession(job.id); job.drawings = s ? [card(s)] : []; }
}

app.post('/api/upload', async (c) => {
  const form = await c.req.parseBody({ all: true });
  const kind = String(form['kind'] ?? '') as Kind;
  const files = ([] as any[]).concat(form['files'] ?? form['file'] ?? []).filter((f) => f && typeof f !== 'string') as File[];

  // routing is manual: the user picks the kind of material, we never sniff it
  if (!KINDS.includes(kind)) return c.json({ error: `kind must be one of ${KINDS.join(', ')}` }, 400);
  if (files.length !== 1) return c.json({ error: 'upload exactly one file' }, 400);
  const f = files[0];
  const name = path.basename(f.name);
  const ext = path.extname(name).toLowerCase();
  if (ext !== EXT_BY_KIND[kind]) {
    return c.json({ error: `kind "${kind}" expects a ${EXT_BY_KIND[kind]} file, got "${ext || name}"` }, 400);
  }
  if (f.size > MAX_MB * 1e6) return c.json({ error: `file is larger than ${MAX_MB} MB` }, 413);

  const buf = Buffer.from(await f.arrayBuffer());
  const uid = createHash('sha256').update(kind).update('\0').update(buf).digest('hex').slice(0, 16);

  // same file again: the index is the cache
  const existing = getSession(uid);
  if (existing) {
    const job: Job = { id: uid, stage: 'ready', log: ['cached'], finished: true, drawings: [card(existing)] };
    jobs.set(uid, job);
    return c.json({ jobId: uid, cached: true });
  }
  if (jobs.get(uid) && !jobs.get(uid)!.finished) return c.json({ jobId: uid });

  // the indexer expects <root>/data/<source>/<files>
  const root = path.join(UPLOADS, uid);
  rmSync(root, { recursive: true, force: true });
  const dir = path.join(root, 'data', SOURCE_BY_KIND[kind]);
  mkdirSync(dir, { recursive: true });
  writeFileSync(path.join(dir, name), buf);

  const job: Job = { id: uid, stage: 'indexing', log: [], finished: false, drawings: [] };
  jobs.set(uid, job);

  const p = spawn(PYTHON, [path.resolve('../indexer/index.py'), root, '--db', path.join(root, 'index.db')]);
  const timer = setTimeout(() => { p.kill(); }, INDEX_TIMEOUT_S * 1000);
  let timedOut = false;
  p.on('exit', (_c, sig) => { if (sig) timedOut = true; });
  p.stdout.on('data', (d) => job.log.push(String(d)));
  p.stderr.on('data', (d) => job.log.push(String(d)));
  p.on('error', (e) => { clearTimeout(timer); finish(job, false, `cannot start ${PYTHON}: ${e.message}`); });
  p.on('close', (code) => {
    clearTimeout(timer);
    if (job.finished) return;
    if (timedOut) return finish(job, false, `parsing took longer than ${INDEX_TIMEOUT_S} s: file is too complex for the limit`);
    if (code !== 0) return finish(job, false, 'indexer failed');
    writeFileSync(metaPath(uid), JSON.stringify({ kind, fileName: name, createdAt: Date.now(), ready: true }));
    const s = getSession(uid);
    if (!s) return finish(job, false, 'nothing was indexed');
    // a structured format that produced no objects failed to parse: say why instead of an empty drawing
    if (kind !== 'png' && elementCount(s.db, s.drawing.id) === 0) {
      const note = (filesOf(s.db, s.drawing.id)[0]?.note as string | undefined) ?? 'no objects found';
      return finish(job, false, `could not parse the file: ${note}`);
    }
    if (kind !== 'pdf') return finish(job, true);
    // vector PDF: draw the sheet so answers can be shown on it; a failure here is not fatal
    job.stage = 'rendering';
    const r = spawn(PYTHON, [path.resolve('../indexer/render_pdf.py'), path.join(dir, name), path.join(root, 'sheet.svg')]);
    const rt = setTimeout(() => r.kill(), INDEX_TIMEOUT_S * 1000);
    r.on('error', () => { clearTimeout(rt); finish(job, true); });
    r.on('close', () => { clearTimeout(rt); finish(job, true); });
  });

  return c.json({ jobId: uid });
});

app.get('/api/jobs/:id', (c) => {
  const job = jobs.get(c.req.param('id'));
  if (!job) return c.json({ error: 'not found' }, 404);
  return streamSSE(c, async (stream) => {
    while (!job.finished) {
      await stream.writeSSE({ data: JSON.stringify(job) });
      await new Promise((r) => setTimeout(r, 500));
    }
    await stream.writeSSE({ data: JSON.stringify(job), event: 'done' });
  });
});

/* ------------------------------------------- static: the built front last */

app.use('/*', serveStatic({ root: './public' }));
app.get('*', serveStatic({ path: './public/index.html' }));

const port = Number(process.env.PORT ?? 3000);
serve({ fetch: app.fetch, port }, () => console.log(`http://localhost:${port}`));
