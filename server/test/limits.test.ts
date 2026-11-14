import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Bucket, Semaphore, isPrivateIp } from '../src/limits.js';

test('this machine and private networks are not limited, public addresses are', () => {
  for (const ip of ['127.0.0.1', '::1', '::ffff:127.0.0.1', '10.1.2.3', '172.17.0.1', '172.31.255.1', '192.168.1.20', 'fd00::1', '169.254.1.1']) {
    assert.equal(isPrivateIp(ip), true, ip);
  }
  for (const ip of ['8.8.8.8', '172.32.0.1', '172.15.0.1', '192.169.0.1', '2606:4700::1111', 'unknown']) {
    assert.equal(isPrivateIp(ip), false, ip);
  }
});

test('a bucket allows a burst of perMin requests, then refills over time', () => {
  const b = new Bucket(6);
  const t0 = 1_000_000;
  for (let i = 0; i < 6; i++) assert.equal(b.take(t0).ok, true);
  const refused = b.take(t0);
  assert.equal(refused.ok, false);
  assert.equal(refused.retryAfter, 10);                           // 6 a minute = one every 10 s
  assert.equal(b.take(t0 + 10_000).ok, true);
  assert.equal(b.take(t0 + 10_000).ok, false);
});

test('a semaphore lets n jobs run and queues the rest', async () => {
  const s = new Semaphore(2);
  await s.acquire(); await s.acquire();
  assert.equal(s.full, true);
  let third = false;
  const p = s.acquire().then(() => { third = true; });
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(third, false);
  s.release();
  await p;
  assert.equal(third, true);
});
