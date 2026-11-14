/**
 * Protection against floods, not against people: nobody logs in, but nobody can drown the server either.
 *
 *  - a token bucket per client address on the routes that cost something (ask, upload);
 *  - requests from this machine or a private network are NOT limited unless RATE_LIMIT_PRIVATE=1, so running the code,
 *    the benchmark (indexer/bench.py) or the Docker image on one's own computer is never slowed down;
 *  - behind a reverse proxy set TRUST_PROXY=<number of proxies, usually 1> so the real address is read from X-Forwarded-For, otherwise every request
 *    looks like the proxy's private address and nothing is limited.
 * Every number is an environment variable; 0 switches that limit off.
 */
import { getConnInfo } from '@hono/node-server/conninfo';
import type { Context, MiddlewareHandler } from 'hono';

// number of reverse proxies in front (0 = none). The client address is the entry that many places from the END of
// X-Forwarded-For: the left side is written by the client and can be forged, the right side is added by our own proxies.
const PROXY_HOPS = Math.max(0, Math.floor(Number(process.env.TRUST_PROXY ?? 0)) || 0);
const LIMIT_PRIVATE = process.env.RATE_LIMIT_PRIVATE === '1';

export const limit = (name: string, fallback: number) => {
  const v = Number(process.env[name] ?? fallback);
  return Number.isFinite(v) && v >= 0 ? v : fallback;
};

export function clientIp(c: Context): string {
  if (PROXY_HOPS > 0) {
    const parts = c.req.header('x-forwarded-for')?.split(',').map((x) => x.trim()).filter(Boolean) ?? [];
    const ip = parts[parts.length - PROXY_HOPS];
    if (ip) return ip;
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

/** At most `perDay` requests per client address per UTC day (counted in memory, reset on restart). Private addresses are exempt. */
export function dailyLimit(name: string, perDay: number): MiddlewareHandler {
  const counts = new Map<string, { day: string; n: number }>();
  return async (c, next) => {
    if (perDay <= 0) return next();
    const ip = clientIp(c);
    if (!LIMIT_PRIVATE && isPrivateIp(ip)) return next();
    const day = new Date().toISOString().slice(0, 10);
    let e = counts.get(ip);
    if (!e || e.day !== day) { if (counts.size > 50_000) counts.clear(); e = { day, n: 0 }; counts.set(ip, e); }
    if (e.n >= perDay) return c.json({ error: `Daily limit reached: ${perDay} ${name} requests per day from one address. It resets at 00:00 UTC.` }, 429);
    e.n++;
    return next();
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
