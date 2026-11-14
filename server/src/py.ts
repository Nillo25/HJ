/** Python interpreter for the indexer. Pin it in deployment: the indexer needs 3.10+. */
export const PYTHON = process.env.PYTHON_BIN ?? (process.platform === 'win32' ? 'python' : 'python3');
