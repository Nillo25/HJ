import assert from 'node:assert/strict';
import { test } from 'node:test';
import { isWeak, marksHistory, planFromScene, toAnswer, withScreenNumbers, type RScene } from '../src/raster.js';

// shapes taken from real responses of the floor-plan service
const scene: RScene = {
  width: 455, height: 649, meta: { scale: { px_per_m: null } },
  objects: [
    { id: 'win1', class: 'window', polygon: [[10, 10], [30, 10], [30, 14], [10, 14]], conf: 0.8, room: 'other_1' },
    { id: 'win2', class: 'window', polygon: [[50, 10], [70, 10], [70, 14], [50, 14]], conf: 0.6, room: 'other_1' },
    { id: 't1', class: 'toilet', polygon: [[100, 100], [110, 100], [110, 112], [100, 112]], conf: 0.9, room: 'bath_1' },
    { id: 'w1', class: 'wall', polygon: [[0, 0], [455, 0], [455, 4], [0, 4]], conf: 0.9, room: null },
  ],
  rooms: [{ id: 'bath_1', class: 'room_bath', polygon: [[90, 90], [140, 90], [140, 140], [90, 140]], conf: 0.7 }],
};

test('a count with low confidence is an estimate, with the windows as evidence and numeric ids', () => {
  const a: any = toAnswer({
    reply: 'Нашёл окна: 2 шт.', confidence: 'low', parsed_by: 'cache', cost_usd: 0,
    calls: [{ tool: 'count', args: { cls: 'window' }, result: { answer: 2, unit: 'шт', ids: ['win1', 'win2'], note: 'проверьте' } }],
  }, scene);
  assert.equal(a.status, 'estimated');
  assert.equal(a.answer, 2);
  assert.equal(a.unit, 'count');
  assert.equal(a.confidence, 0.5);
  assert.equal(a.route, 'template');
  assert.deepEqual(a.evidence.map((e: any) => [e.id, e.ref, e.type]), [[1, 'win1', 'window'], [2, 'win2', 'window']]);
  assert.deepEqual(JSON.parse(a.evidence[0].bbox), [10, 10, 30, 14]);
  assert.match(a.note, /проверьте/);
});

test('a detection under its class threshold + margin is weak (thresholds.json): drawn yellow', () => {
  assert.equal(isWeak('window', 0.55), true);        // window: 0.45 + 0.15
  assert.equal(isWeak('window', 0.8), false);
  assert.equal(isWeak('door', 0.52), false);         // door: 0.35 + 0.15
  assert.equal(isWeak('toilet', 0.32), true);        // toilet: 0.25 + 0.15
  const a: any = toAnswer({
    reply: 'ok', confidence: 'low',
    calls: [{ tool: 'count', args: { cls: 'window' }, result: { answer: 2, unit: 'шт', ids: ['win1', 'win2'] } }],
  }, { ...scene, objects: scene.objects.map(o => (o.id === 'win2' ? { ...o, conf: 0.5 } : o)) });
  assert.deepEqual(a.evidence.map((e: any) => [e.ref, e.weak]), [['win1', false], ['win2', true]]);
});

test('a highlight with high confidence is answered; ids come from the action too', () => {
  const a: any = toAnswer({
    reply: 'Подсветил унитазы: 1 шт.', confidence: 'high', action: { type: 'highlight', ids: ['t1'] },
    calls: [{ tool: 'highlight', result: { answer: 1, unit: 'шт', ids: ['t1'] } }],
  }, scene);
  assert.equal(a.status, 'answered');
  assert.equal(a.confidence, 0.95);
  assert.equal(a.evidence.length, 1);
  assert.equal(a.evidence[0].id, 3);
});

test('paid calls make the route "llm" and the unit without a scale is px²', () => {
  const a: any = toAnswer({
    reply: 'Площадь: 1200', confidence: 'medium', cost_usd: 0.0005,
    calls: [{ tool: 'area', result: { answer: 1200, unit: 'px2', ids: ['bath_1'] } }],
  }, scene);
  assert.equal(a.route, 'llm');
  assert.equal(a.unit, 'px²');
  assert.equal(a.evidence[0].type, 'room');
  assert.equal(a.evidence[0].subtype, 'bath');
});

test('a list answer (rooms) is shown as the reply text', () => {
  const a: any = toAnswer({
    reply: 'Комнаты: bath_1', confidence: 'low',
    calls: [{ tool: 'list_rooms', result: { answer: [{ id: 'bath_1' }], ids: ['bath_1'] } }],
  }, scene);
  assert.equal(a.answer, 'Комнаты: bath_1');
  assert.equal(a.unit, undefined);
});

test('nothing found and nothing understood are refusals with the service text as the reason', () => {
  const none: any = toAnswer({ reply: 'Комнату не нашёл.', confidence: 'low', calls: [{ tool: 'area', result: { answer: null, ids: [] } }] }, scene);
  assert.equal(none.status, 'refused');
  assert.equal(none.reason, 'Комнату не нашёл.');
  const lost: any = toAnswer({ reply: 'Не понял запрос', parsed_by: 'none', calls: [] }, scene);
  assert.equal(lost.status, 'refused');
});

test('the plan draws the scene over the image: layers by kind, ids shared with the evidence', () => {
  const p: any = planFromScene(scene, '/img');
  assert.deepEqual(p.viewBox, [0, 0, 455, 649]);
  assert.equal(p.background, '/img');
  assert.equal(p.overlay, true);
  assert.deepEqual(Object.keys(p.layers).sort(), ['doors', 'fixtures', 'rooms', 'walls', 'windows'].filter((k) => k !== 'doors'));
  assert.equal(p.layers.windows.length, 2);
  const ev: any = toAnswer({ reply: 'x', confidence: 'high', calls: [{ tool: 'count', result: { answer: 1, ids: ['t1'] } }] }, scene);
  assert.equal(p.layers.fixtures[0].id, ev.evidence[0].id);
});

test('the next question carries the numbers on screen: number = element id, so "no window 2" means win2', () => {
  assert.deepEqual(marksHistory(undefined), []);
  assert.deepEqual(marksHistory({ question: 'q', reply: 'r', refs: [] }), []);
  const h = marksHistory({ question: 'how many windows', reply: '2 windows', refs: ['win1', 'win7'] });
  assert.equal(h.length, 2);
  assert.deepEqual(h[0], { role: 'user', content: 'how many windows' });
  assert.equal(h[1].role, 'assistant');
  assert.match(h[1].content, /1=win1, 2=win7/);
});

test('a number the person sees becomes the element id in the message; a plain count does not', () => {
  const refs = Array.from({ length: 24 }, (_, i) => `win${i + 2}`);      // the screen's 18 is win19
  assert.match(withScreenNumbers('тут нет окна номер 18', refs), /18 = element win19/);
  assert.match(withScreenNumbers('window 18 is wrong', refs), /18 = element win19/);
  assert.match(withScreenNumbers('remove #3 and №5', refs), /3 = element win4, 5 = element win6/);
  assert.equal(withScreenNumbers('на плане 12 окон', refs), 'на плане 12 окон');
  assert.equal(withScreenNumbers('there are 12 windows', refs), 'there are 12 windows');
  assert.equal(withScreenNumbers('window 99', refs), 'window 99');            // not on screen: left to the model
  assert.equal(withScreenNumbers('номер 18', undefined), 'номер 18');
});
