import { serve } from '@hono/node-server';
import { serveStatic } from '@hono/node-server/serve-static';
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { streamSSE } from 'hono/streaming';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { Readable } from 'node:stream';
import path from 'node:path';

import {
  db, listDrawings, drawingById, filesOf, floorsOf, queryElements, latestEval,
  type Element,
} from './db.js';
import { ask } from './ask.js';
import { countQuestion, usageSnapshot } from './usage.js';

const app = new Hono();
app.use('/api/*', cors());

/* ------------------------------------------------------------ чертежи */

app.get('/api/drawings', (c) => {
  const limit = Number(c.req.query('limit') ?? 500);
  const rows = (listDrawings.all(limit) as any[]).map((r) => ({
    ...r,
    capabilities: r.capabilities ? JSON.parse(r.capabilities) : null,
  }));
  return c.json({ drawings: rows });
});

app.get('/api/drawings/:id', (c) => {
  const id = Number(c.req.param('id'));
  const d = drawingById.get(id) as any;
  if (!d) return c.json({ error: 'not found' }, 404);
  return c.json({
    ...d,
    capabilities: d.capabilities ? JSON.parse(d.capabilities) : null,
    files: filesOf.all(id),
    floors: floorsOf.all(id),
    viewer: viewerInfo(id),
    counts: db.prepare(
      'SELECT type, subtype, COUNT(*) n FROM elements WHERE drawing_id=? GROUP BY type, subtype',
    ).all(id),
  });
});

/* ------------------------------------------- исходный файл для 3D-просмотрщика */

const VIEWER_MAX_MB = Number(process.env.VIEWER_MAX_MB ?? 100);
const ALLOWED_ROOTS = [path.resolve('uploads'), process.env.DATASET_ROOT]
  .filter(Boolean).map((p) => path.resolve(p as string));

function sourceFile(drawingId: number, format = 'ifc') {
  const f = (filesOf.all(drawingId) as any[]).find((x) => x.format === format);
  if (!f) return null;
  const abs = path.resolve(f.path);
  // serve only what we indexed ourselves
  if (!ALLOWED_ROOTS.some((r) => abs.startsWith(r + path.sep)) || !existsSync(abs)) return null;
  return { id: f.id, abs, mb: statSync(abs).size / 1e6 };
}

function viewerInfo(drawingId: number) {
  const f = sourceFile(drawingId, 'ifc');
  return f ? { format: 'ifc', file_id: f.id, size_mb: Math.round(f.mb * 10) / 10,
               allowed: f.mb <= VIEWER_MAX_MB } : null;
}

app.get('/api/drawings/:id/source', (c) => {
  const f = sourceFile(Number(c.req.param('id')));
  if (!f) return c.json({ error: 'no source file' }, 404);
  if (f.mb > VIEWER_MAX_MB) return c.json({ error: 'file too large for the viewer' }, 413);
  return c.body(Readable.toWeb(createReadStream(f.abs)) as ReadableStream, 200, {
    'content-type': 'application/octet-stream',
    'content-length': String(statSync(f.abs).size),
  });
});

/* ------------------------------- геометрия для просмотрщика (см. §6 контракта) */

app.get('/api/drawings/:id/plan', (c) => {
  const id = Number(c.req.param('id'));
  const els = queryElements.all({
    drawing_id: id, type: null, subtype: null, floor: null, literal: null,
  }) as Element[];
  if (!els.length) return c.json({ error: 'no geometry' }, 404);

  // общий габарит -> экранный viewBox 1200 x N, ось Y переворачивается
  let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const e of els) {
    if (!e.bbox) continue;
    const b = JSON.parse(e.bbox);
    x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]);
    x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]);
  }
  const W = 1200;
  const s = W / Math.max(x1 - x0, 1e-6);
  const H = Math.round((y1 - y0) * s);
  const tx = ([x, y]: number[]) => [
    Math.round((x - x0) * s * 10) / 10,
    Math.round((y1 - y) * s * 10) / 10,     // flip Y: в чертеже вверх, на экране вниз
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

/* --------------------------------------------------------------- вопрос */

app.post('/api/ask', async (c) => {
  const body = await c.req.json<{ question: string; drawing?: string; drawing_id?: number }>();
  if (!body?.question) return c.json({ error: 'question required' }, 400);

  let name = body.drawing;
  if (!name && body.drawing_id) {
    name = (drawingById.get(body.drawing_id) as any)?.name;
  }
  if (!name) return c.json({ error: 'drawing required' }, 400);

  try {
    const res: any = await ask(name, body.question);
    countQuestion(res.route === 'template' || res.route === 'cache');
    return c.json(res);
  } catch (e: any) {
    return c.json({ status: 'refused', reason: `${e?.name}: ${e?.message}`,
                    missing: 'internal', route: 'llm', cost_usd: 0 }, 200);
  }
});

/* -------------------------------------------------------- панель точности */

app.get('/api/usage', (c) => c.json(usageSnapshot()));

app.get('/api/eval/latest', (c) => c.json({ rows: latestEval.all() }));

/* ------------------------------------------------- загрузка и индексация */

type Job = { id: string; stage: string; done: number; total: number; log: string[]; finished: boolean; drawings: any[] };
const jobs = new Map<string, Job>();

app.post('/api/upload', async (c) => {
  const form = await c.req.parseBody({ all: true });
  const files = ([] as any[]).concat(form['files'] ?? form['file'] ?? []);
  if (!files.length) return c.json({ error: 'no files' }, 400);

  const id = randomUUID().slice(0, 8);
  // the indexer expects <root>/data/<source>/<files>; each upload is its own source
  const root = path.resolve('uploads', id);
  const dir = path.join(root, 'data', id);
  mkdirSync(dir, { recursive: true });

  const stems: string[] = [];
  for (const f of files as File[]) {
    const name = path.basename(f.name);                 // no path tricks from the client
    writeFileSync(path.join(dir, name), Buffer.from(await f.arrayBuffer()));
    stems.push(name.replace(/\.[^.]+$/, ''));
  }

  const job: Job = { id, stage: 'indexing', done: 0, total: files.length, log: [], finished: false, drawings: [] };
  jobs.set(id, job);

  // the indexer is a separate Python process, once per upload
  const p = spawn('python3', [path.resolve('../indexer/index.py'), root, '--db', process.env.DB_PATH ?? '../app.db']);
  p.stdout.on('data', (d) => { job.log.push(String(d)); });
  p.stderr.on('data', (d) => { job.log.push(String(d)); });
  p.on('close', (code) => {
    job.finished = true;
    job.stage = code === 0 ? 'ready' : 'failed';
    if (code === 0) {
      const names = [...new Set(stems)];
      job.drawings = db.prepare(
        `SELECT d.id, d.name, d.source, d.capabilities,
                (SELECT COUNT(*) FROM elements e WHERE e.drawing_id = d.id) AS elements
           FROM drawings d WHERE d.name IN (${names.map(() => '?').join(',')})`,
      ).all(...names).map((r: any) => ({ ...r, capabilities: r.capabilities ? JSON.parse(r.capabilities) : null }));
    }
  });

  return c.json({ jobId: id });
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

/* ------------------------------------------- статика: собранный фронт последним */

app.use('/*', serveStatic({ root: './public' }));
app.get('*', serveStatic({ path: './public/index.html' }));

const port = Number(process.env.PORT ?? 3000);
serve({ fetch: app.fetch, port });
console.log(`http://localhost:${port}`);
