PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS drawings (
  id            INTEGER PRIMARY KEY,
  name          TEXT NOT NULL UNIQUE,
  source        TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'pending',
  capabilities  TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS files (
  id          INTEGER PRIMARY KEY,
  drawing_id  INTEGER NOT NULL REFERENCES drawings(id),
  path        TEXT NOT NULL,
  sha256      TEXT NOT NULL UNIQUE,
  format      TEXT NOT NULL,
  role        TEXT NOT NULL,
  width       REAL,
  height      REAL,
  transform   TEXT,
  status      TEXT NOT NULL DEFAULT 'ok',
  note        TEXT
);

CREATE TABLE IF NOT EXISTS floors (
  id          INTEGER PRIMARY KEY,
  drawing_id  INTEGER NOT NULL REFERENCES drawings(id),
  file_id     INTEGER REFERENCES files(id),
  name        TEXT,
  level       REAL,
  ord         INTEGER
);

CREATE TABLE IF NOT EXISTS elements (
  id          INTEGER PRIMARY KEY,
  drawing_id  INTEGER NOT NULL REFERENCES drawings(id),
  file_id     INTEGER NOT NULL REFERENCES files(id),
  floor_id    INTEGER REFERENCES floors(id),
  type        TEXT NOT NULL,
  subtype     TEXT,
  label       TEXT,
  label_norm  TEXT,
  area        REAL,
  length      REAL,
  volume      REAL,
  bbox        TEXT,
  polygon     TEXT,
  page        INTEGER,
  source_ref  TEXT,
  confidence  REAL DEFAULT 1.0,
  is_primary  INTEGER DEFAULT 1,
  props       TEXT
);
CREATE INDEX IF NOT EXISTS idx_el_lookup ON elements(drawing_id, type, subtype, floor_id);
CREATE INDEX IF NOT EXISTS idx_el_label  ON elements(drawing_id, label_norm);

CREATE TABLE IF NOT EXISTS text_spans (
  id         INTEGER PRIMARY KEY,
  file_id    INTEGER NOT NULL REFERENCES files(id),
  page       INTEGER,
  text       TEXT NOT NULL,
  bbox       TEXT NOT NULL,
  confidence REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS label_dict (
  label_norm TEXT PRIMARY KEY,
  subtype    TEXT NOT NULL,
  source     TEXT NOT NULL DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS question_cache (
  id       INTEGER PRIMARY KEY,
  template TEXT NOT NULL UNIQUE,
  fn       TEXT NOT NULL,
  args     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
  id         INTEGER PRIMARY KEY,
  run_id     TEXT NOT NULL,
  path       TEXT NOT NULL,
  question   TEXT NOT NULL,
  expected   TEXT,
  got        TEXT,
  verdict    TEXT NOT NULL,
  q_type     TEXT,
  source     TEXT,
  route      TEXT,
  cost_usd   REAL,
  latency_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_eval_run ON eval_results(run_id, source, q_type);
