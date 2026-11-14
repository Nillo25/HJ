// One command for local development: API (server, port 3000) + web (Vite, port 5173). Node 22+, no dependencies.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const win = process.platform === 'win32';
const npm = win ? 'npm.cmd' : 'npm';

function run(cmd, args, cwd, opts = {}) {
  return spawnSync(cmd, args, { cwd, stdio: 'inherit', shell: win, ...opts });
}

// dependencies, once
for (const dir of ['server', 'web']) {
  if (!existsSync(path.join(root, dir, 'node_modules'))) {
    console.log(`[dev] npm install in ${dir}/`);
    if (run(npm, ['install'], path.join(root, dir)).status !== 0) process.exit(1);
  }
}

// Python 3.10+ for the indexer: PYTHON_BIN, else the first suitable interpreter on PATH
function pythonOk(bin) {
  const r = spawnSync(bin, ['-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'], { stdio: 'ignore' });
  return r.status === 0;
}
const candidates = [process.env.PYTHON_BIN, 'python3.13', 'python3.12', 'python3.11', 'python3.10', 'python3', 'python'].filter(Boolean);
const python = candidates.find(pythonOk);
if (!python) {
  console.error('[dev] Python 3.10+ not found. Install it (macOS: brew install python@3.12) or set PYTHON_BIN=/path/to/python.');
  process.exit(1);
}
const hasPdf = spawnSync(python, ['-c', 'import pdfminer'], { stdio: 'ignore' }).status === 0;
if (!hasPdf) console.warn(`[dev] pdfminer.six is missing for ${python}: PDF text will not be read. Fix: ${python} -m pip install -r indexer/requirements.txt`);
if (!existsSync(path.join(root, 'server', '.env'))) console.warn('[dev] server/.env not found: only questions that need no model will work (see server/.env.example).');
console.log(`[dev] python: ${python}\n[dev] open http://localhost:5173  (API on :3000)\n`);

// The floor-plan service for PNG (raster/): started when its weights and Python packages are there (see raster/WEIGHTS.md).
// RASTER_PYTHON is the interpreter with ultralytics installed (default: the one above); RASTER_URL set = an existing service is used.
const rasterPython = process.env.RASTER_PYTHON || python;
const rasterDir = path.join(root, 'raster');
const hasWeights = existsSync(path.join(rasterDir, 'runs', 'colab', 'floorplan_v2.pt')) || Boolean(process.env.FP_MODEL);
const rasterReady = !process.env.RASTER_URL && hasWeights &&
  spawnSync(rasterPython, ['-c', 'import ultralytics, fastapi, uvicorn'], { stdio: 'ignore' }).status === 0;
if (process.env.RASTER_URL) console.log(`[dev] floor-plan service: ${process.env.RASTER_URL}`);
else if (!rasterReady) console.warn('[dev] floor-plan service not started (needs raster/runs/colab/floorplan_v2.pt and `pip install -r raster/requirements.txt`): PNG plans are only shown, not read.');
else console.log('[dev] floor-plan service: http://localhost:8001');

const env = { ...process.env, PYTHON_BIN: python, ...(rasterReady ? { RASTER_URL: 'http://localhost:8001' } : {}) };
const procs = [
  ['api', spawn(npm, ['run', 'dev'], { cwd: path.join(root, 'server'), env, shell: win })],
  ['web', spawn(npm, ['run', 'dev'], { cwd: path.join(root, 'web'), env, shell: win })],
  ...(rasterReady ? [['raster', spawn(rasterPython, ['-m', 'uvicorn', 'server.app:app', '--port', '8001'], { cwd: rasterDir, env, shell: false })]] : []),
];
const tag = (name, stream, out) => {
  let buf = '';
  stream.on('data', (d) => {
    buf += d;
    const lines = buf.split(/\r?\n/); buf = lines.pop();
    for (const l of lines) out.write(`[${name}] ${l}\n`);
  });
};
let stopping = false;
const stop = (code = 0) => {
  if (stopping) return; stopping = true;
  for (const [, p] of procs) {
    if (win) spawnSync('taskkill', ['/pid', String(p.pid), '/t', '/f'], { stdio: 'ignore' });
    else p.kill('SIGTERM');
  }
  setTimeout(() => process.exit(code), 300);
};
for (const [name, p] of procs) {
  tag(name, p.stdout, process.stdout); tag(name, p.stderr, process.stderr);
  p.on('exit', (c) => { if (!stopping) { console.error(`[dev] ${name} stopped (code ${c}), stopping the other one`); stop(c ?? 1); } });
}
process.on('SIGINT', () => stop(0));
process.on('SIGTERM', () => stop(0));
