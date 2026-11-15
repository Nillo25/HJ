/**
 * Python interpreter for the indexer (needs 3.10+). PYTHON_BIN wins; otherwise the first suitable one on PATH is used,
 * because the system python3 on macOS is 3.9 and would fail inside the indexer.
 */
import { spawnSync } from 'node:child_process';

const ok = (bin: string) =>
  spawnSync(bin, ['-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'], { stdio: 'ignore' }).status === 0;

function pick(): string {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  const names = process.platform === 'win32'
    ? ['python', 'py']
    : ['python3', 'python3.13', 'python3.12', 'python3.11', 'python3.10', 'python'];
  const found = names.find(ok);
  if (!found) console.error('Python 3.10+ not found. Install it (macOS: brew install python@3.12) or set PYTHON_BIN=/path/to/python.');
  return found ?? names[0];
}

export const PYTHON = pick();
console.log(`python: ${PYTHON}`);
