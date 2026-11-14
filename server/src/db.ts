import Database from 'better-sqlite3';

export const db = new Database(process.env.DB_PATH ?? '../app.db', {
  readonly: false,
  fileMustExist: true,
});
db.pragma('journal_mode = WAL');

export type Element = {
  id: number;
  drawing_id: number;
  file_id: number;
  floor_id: number | null;
  type: string;
  subtype: string | null;
  label: string | null;
  label_norm: string | null;
  area: number | null;
  length: number | null;
  volume: number | null;
  bbox: string | null;
  polygon: string | null;
  page: number | null;
  source_ref: string | null;
  confidence: number;
  is_primary: number;
  props: string | null;
};

export const drawingByName = db.prepare(
  'SELECT * FROM drawings WHERE name = ?',
);

export const drawingById = db.prepare('SELECT * FROM drawings WHERE id = ?');

export const listDrawings = db.prepare(
  `SELECT d.id, d.name, d.source, d.status, d.capabilities,
          (SELECT COUNT(*) FROM elements e WHERE e.drawing_id = d.id) AS elements
   FROM drawings d ORDER BY d.source, d.name LIMIT ?`,
);

export const filesOf = db.prepare('SELECT * FROM files WHERE drawing_id = ?');

export const floorsOf = db.prepare(
  'SELECT * FROM floors WHERE drawing_id = ? ORDER BY ord',
);

/** Элементы с фильтрами. NULL-аргумент = фильтр не применяется. */
export const queryElements = db.prepare(
  `SELECT e.* FROM elements e
    WHERE e.drawing_id = @drawing_id
      AND (@type     IS NULL OR e.type    = @type)
      AND (@subtype  IS NULL OR e.subtype = @subtype)
      AND (@floor    IS NULL OR e.floor_id = (
            SELECT id FROM floors
             WHERE drawing_id = @drawing_id AND name = @floor))
      AND (@literal  IS NULL OR LOWER(e.label) LIKE '%' || LOWER(@literal) || '%')
      AND e.is_primary = 1`,
);

export const cacheGet = db.prepare(
  'SELECT fn, args FROM question_cache WHERE template = ?',
);

export const cachePut = db.prepare(
  'INSERT OR IGNORE INTO question_cache(template, fn, args) VALUES (?, ?, ?)',
);

export const labelDict = db.prepare('SELECT label_norm, subtype FROM label_dict');

export const latestEval = db.prepare(
  `SELECT source, q_type, verdict, COUNT(*) AS n,
          AVG(cost_usd) AS avg_cost, AVG(latency_ms) AS avg_ms
     FROM eval_results
    WHERE run_id = (SELECT run_id FROM eval_results ORDER BY id DESC LIMIT 1)
    GROUP BY source, q_type, verdict`,
);
