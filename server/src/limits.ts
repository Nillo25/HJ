/**
 * Protection against floods, not against people: nobody logs in, but nobody can drown the server either.
 *
 *  - a token bucket per client address on the routes that cost something (ask, upload);
 *  - requests from this machine or a private network are NOT limited unless RATE_LIMIT_PRIVATE=1, so running the code,
 *    the benchmark (indexer/bench.py) or the Docker image on one's own computer is never slowed down;
 *  - behind a reverse proxy set TRUST_PROXY=1 so the real address is read from X-Forwarded-For, otherwise every request
 *    looks like the proxy's private address and nothing is limited.
 * Every number is an environment variable; 0 switches that limit off.
 */
import { getConnInfo } from '@hono/node-server/conninfo';
import type { Context, MiddlewareHandler } from 'hono';

const TRUST_PROXY = process.env.TRUST_PROXY === '1';
const LIMIT_PRIVATE = process.env.RATE_LIMIT_PRIVATE === '1';

export const limit = (name: string, fallback: number) => {
  const v = Number(process.env[name] ?? fallback);
  return Number.isFinite(v) && v >= 0 ? v : fallback;
};

export function clientIp(c: Context): string {
  if (TRUST_PROXY) {
    const xff = c.req.header('x-forwarded-for')?.split(',')[0]?.trim();
    if (xff) return xff;
  }
  try { return getConnInfo(c).remote.address ?? 'unknown'; } catch { return 'unknown'; }
}

/** Loopback and private networks (also IPv4-mapped IPv6 such as ::ffff:127.0.0.1). */
export function isPrivateIp(ip: string): boolean {
  const a = ip.replace(/^::ffff:/i, '').toLowerCase();
  if (a === '::1' || a === 'localhost') return true;
  if (/^(fc|fd)[0-9a-f]{2}:/.test(a) || a.startsWith('fe80:')) return true;          // IPv6 unique-local, link-local
  const m = a.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
  if (!m) return false;
  const [o1, o2] = [Number(m[1]), Number(m[2])];
  return o1 === 127 || o1 === 10 || (o1 === 172 && o2 >= 16 && o2 <= 31) || (o1 === 192 && o2 === 168) || (o1 === 169 && o2 === 254);
}

/** Token bucket: `perMin` tokens a minute, a burst of up to `perMin` at once. */
export class Bucket {
  private tokens: number;
  private at: number | null = null;
  constructor(private perMin: number) { this.tokens = perMin; }
  take(now = Date.now()): { ok: boolean; retryAfter: number } {
    if (this.at === null) this.at = now;
    this.tokens = Math.min(this.perMin, this.tokens + ((now - this.at) / 60_000) * this.perMin);
    this.at = now;
    if (this.tokens >= 1) { this.tokens -= 1; return { ok: true, retryAfter: 0 }; }
    return { ok: false, retryAfter: Math.ceil(((1 - this.tokens) / this.perMin) * 60) };
  }
  idle(now = Date.now()) { return this.at !== null && now - this.at > 10 * 60_000; }
}

export function rateLimit(name: string, perMin: number): MiddlewareHandler {
  const buckets = new Map<string, Bucket>();
  if (perMin > 0) {
    setInterval(() => { for (const [k, b] of buckets) if (b.idle()) buckets.delete(k); }, 5 * 60_000).unref();
  }
  return async (c, next) => {
    if (perMin <= 0) return next();
    const ip = clientIp(c);
    if (!LIMIT_PRIVATE && isPrivateIp(ip)) return next();
    let b = buckets.get(ip);
    if (!b) { b = new Bucket(perMin); buckets.set(ip, b); }
    const r = b.take();
    if (r.ok) return next();
    c.header('Retry-After', String(r.retryAfter));
    return c.json({ error: `Too many ${name} requests: wait ${r.retryAfter} s and try again.` }, 429);
  };
}

/** Lets `n` jobs run at once; the rest wait their turn. */
export class Semaphore {
  private waiting: (() => void)[] = [];
  private busy = 0;
  constructor(private n: number) {}
  async acquire() {
    if (this.n <= 0 || this.busy < this.n) { this.busy++; return; }
    await new Promise<void>((res) => this.waiting.push(res));
    this.busy++;
  }
  release() { this.busy--; this.waiting.shift()?.(); }
  get full() { return this.n > 0 && this.busy >= this.n; }
}
