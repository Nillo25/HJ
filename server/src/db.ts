/**
 * Per-upload databases. There is no dataset-wide database in the product:
 * every upload is parsed into uploads/<uid>/index.db, and the uid is a hash of
 * the kind and the file contents, so the same file is never parsed twice.
 * The only shared database is cache.db (question template -> parsed call).
 */
import Database from 'better-sqlite3';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

export type Kind = 'dxf' | 'ifc' | 'pdf' | 'png';
export const KINDS: Kind[] = ['dxf', 'ifc', 'pdf', 'png'];

/** The indexer picks per-source logic from the folder name; the kind decides it here. */
export const SOURCE_BY_KIND: Record<Kind, string> = {
  dxf: 'swiss_dwellings', ifc: 'ifc_models', pdf: 'opening_design', png: 'cubicasa5k',
};

export const UPLOADS = path.resolve(process.env.UPLOADS_DIR ?? 'uploads');
const UPLOAD_TTL_H = Number(process.env.UPLOAD_TTL_HOURS ?? 24);
/**
 * Hash of the indexer sources. It is part of every upload id, so an index built by an
 * older parser is never reused: changing the parser invalidates the cache by itself.
 */
export const INDEX_VERSION = (() => {
  const dir = path.resolve('../indexer');
  const h = createHash('sha256');
  try {
    for (const f of readdirSync(dir).filter((n) => n.endsWith('.py') || n.endsWith('.sql')).sort()) {
      h.update(f).update(readFileSync(path.join(dir, f)));
    }
    // the label dictionaries decide what a room is: a changed entry must not be served from an old index
    const dd = path.join(dir, 'dict');
    for (const f of readdirSync(dd).filter((n) => n.endsWith('.json')).sort()) h.update(f).update(readFileSync(path.join(dd, f)));
  } catch { /* no indexer next to the server: version stays constant */ }
  return h.digest('hex').slice(0, 12);
})();

const UID_RE = /^[0-9a-f]{16}$/;

export type Element = {
  id: number; drawing_id: number; file_id: number; floor_id: number | null;
  type: string; subtype: string | null; label: string | null; label_norm: string | null;
  area: number | null; length: number | null; volume: number | null;
  bbox: string | null; polygon: string | null; page: number | null;
  source_ref: string | null; confidence: number; is_primary: number; props: string | null;
};

type Meta = { kind: Kind; fileName: string; createdAt: number; ready: boolean; indexVersion?: string };
export type Session = {
  uid: string; dir: string; dbPath: string; db: Database.Database; meta: Meta;
  drawing: { id: number; name: string; source: string; status: string; capabilities: any };
};

const open = new Map<string, Session>();

export const metaPath = (uid: string) => path.join(UPLOADS, uid, 'meta.json');

function readMeta(uid: string): Meta | null {
  try { return JSON.parse(readFileSync(metaPath(uid), 'utf8')); } catch { return null; }
}

/** Open (or return the open) session; null when the upload is unknown or not indexed. */
export function getSession(uid: string): Session | null {
  if (!UID_RE.test(uid)) return null;
  const hit = open.get(uid);
  if (hit) return hit;
  const meta = readMeta(uid);
  const dbPath = path.join(UPLOADS, uid, 'index.db');
  // an index built by an older parser or dictionary is stale: it must not be opened, the file has to be parsed again
  if (!meta?.ready || meta.indexVersion !== INDEX_VERSION || !existsSync(dbPath)) return null;
  const db = new Database(dbPath, { fileMustExist: true });
  db.pragma('journal_mode = WAL');
  const d = db.prepare('SELECT * FROM drawings ORDER BY id LIMIT 1').get() as any;
  if (!d) { db.close(); return null; }
  const s: Session = {
    uid, dir: path.join(UPLOADS, uid), dbPath, db, meta,
    drawing: { id: d.id, name: d.name, source: d.source, status: d.status,
               capabilities: d.capabilities ? JSON.parse(d.capabilities) : null },
  };
  open.set(uid, s);
  return s;
}

export function listSessions(): Session[] {
  if (!existsSync(UPLOADS)) return [];
  return readdirSync(UPLOADS).filter((u) => UID_RE.test(u))
    .map((u) => getSession(u)).filter((s): s is Session => !!s && s.meta.indexVersion === INDEX_VERSION)
    .sort((a, b) => b.meta.createdAt - a.meta.createdAt);
}

/** Delete uploads older than UPLOAD_TTL_HOURS (0 = keep everything). */
export function cleanupUploads() {
  if (!UPLOAD_TTL_H || !existsSync(UPLOADS)) return;
  const limit = Date.now() - UPLOAD_TTL_H * 3600_000;
  for (const u of readdirSync(UPLOADS)) {
    if (!UID_RE.test(u)) continue;
    const dir = path.join(UPLOADS, u);
    try {
      if (statSync(dir).mtimeMs < limit) {
        open.get(u)?.db.close(); open.delete(u);
        rmSync(dir, { recursive: true, force: true });
      }
    } catch { /* ignore */ }
  }
}

export function ensureUploads() { mkdirSync(UPLOADS, { recursive: true }); }

const dirBytes = (dir: string): number => {
  let n = 0;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    try { n += e.isDirectory() ? dirBytes(p) : statSync(p).size; } catch { /* vanished while counting */ }
  }
  return n;
};

/**
 * Disk quota for uploads (UPLOADS_MAX_GB, 0 = none): before a new upload is stored the oldest ones are removed
 * until it fits. Uploads being parsed right now are never removed.
 */
export function enforceQuota(extraBytes: number, busy: Set<string>) {
  const cap = Number(process.env.UPLOADS_MAX_GB ?? 5) * 1e9;
  if (!(cap > 0) || !existsSync(UPLOADS)) return;
  const dirs = readdirSync(UPLOADS).filter((u) => UID_RE.test(u)).map((u) => {
    const dir = path.join(UPLOADS, u);
    return { u, dir, at: statSync(dir).mtimeMs, size: dirBytes(dir) };
  }).sort((a, b) => a.at - b.at);
  let total = dirs.reduce((n, x) => n + x.size, 0) + extraBytes;
  for (const x of dirs) {
    if (total <= cap) break;
    if (busy.has(x.u)) continue;
    open.get(x.u)?.db.close(); open.delete(x.u);
    rmSync(x.dir, { recursive: true, force: true });
    total -= x.size;
  }
}

/* ------------------------------------------------------ queries per session */

export const filesOf = (db: Database.Database, id: number) =>
  db.prepare('SELECT * FROM files WHERE drawing_id = ?').all(id) as any[];

export const floorsOf = (db: Database.Database, id: number) =>
  db.prepare('SELECT * FROM floors WHERE drawing_id = ? ORDER BY ord').all(id) as any[];

export const elementCount = (db: Database.Database, id: number) =>
  (db.prepare('SELECT COUNT(*) n FROM elements WHERE drawing_id = ?').get(id) as any).n as number;

/** Filters; a null argument means no filter. */
export function queryElements(db: Database.Database, a: {
  drawing_id: number; type: string | null; subtype: string | null;
  floor: string | null; literal: string | null;
}): Element[] {
  return db.prepare(
    `SELECT e.* FROM elements e
      WHERE e.drawing_id = @drawing_id
        AND (@type     IS NULL OR e.type    = @type)
        AND (@subtype  IS NULL OR e.subtype = @subtype)
        AND (@floor    IS NULL OR e.floor_id = (
              SELECT id FROM floors
               WHERE drawing_id = @drawing_id AND LOWER(name) = LOWER(@floor)))
        AND (@literal  IS NULL OR LOWER(e.label) LIKE '%' || LOWER(@literal) || '%')
        AND e.is_primary = 1`,
  ).all(a) as Element[];
}

/* --------------------------------------- the one shared database: the cache */

const cachePath = process.env.CACHE_DB ?? 'cache.db';
if (cachePath !== ':memory:') mkdirSync(path.dirname(path.resolve(cachePath)), { recursive: true });
const cacheDb = new Database(cachePath);
cacheDb.pragma('journal_mode = WAL');
// version 2: entries are templates with {i} slots. Version 1 stored constants of the first question
// that was asked and answered every similar question with them: those rows must not be reused.
const CACHE_VERSION = 3;      // 3: floor_area exists, earlier parses of "total area of the floor" were sum_quantity
if ((cacheDb.pragma('user_version', { simple: true }) as number) !== CACHE_VERSION) {
  cacheDb.exec('DROP TABLE IF EXISTS question_cache');
  cacheDb.pragma(`user_version = ${CACHE_VERSION}`);
}
cacheDb.exec(`CREATE TABLE IF NOT EXISTS question_cache (
  template TEXT PRIMARY KEY, fn TEXT NOT NULL, args TEXT NOT NULL)`);
const cacheGetStmt = cacheDb.prepare('SELECT fn, args FROM question_cache WHERE template = ?');
const cachePutStmt = cacheDb.prepare('INSERT OR IGNORE INTO question_cache(template, fn, args) VALUES (?, ?, ?)');
export const cacheGet = (t: string) => cacheGetStmt.get(t) as { fn: string; args: string } | undefined;
export const cachePut = (t: string, fn: string, args: string) => cachePutStmt.run(t, fn, args);
