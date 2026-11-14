// The question cache once stored the constants of the first question and answered every similar one with them.
// These tests pin the two functions that make a parsed question reusable.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { abstractArgs, templatize } from '../src/ask.js';

test('a quoted label and a number become slots, digits inside the label stay in it', () => {
  const { template, slots } = templatize('Kui suur on ruumi „TUALETTRUUM (24-3)“ korrusel „2.korrus“?');
  assert.equal(template, 'kui suur on ruumi {0} korrusel {1}');
  assert.deepEqual(slots, ['tualettruum (24-3)', '2.korrus']);
});

test('numbers outside quotes are slots too', () => {
  const { template, slots } = templatize('How many windows on floor 3?');
  assert.equal(template, 'how many windows on floor {0}');
  assert.deepEqual(slots, ['3']);
});

test('values taken from the question become slots', () => {
  const q = 'Kui suur on ruumi „TUALETTRUUM (24-3)“ korrusel „2.korrus“?';
  const { slots } = templatize(q);
  assert.deepEqual(abstractArgs({ label: 'TUALETTRUUM (24-3)', floor: '2.korrus' }, slots, q), { label: '{0}', floor: '{1}' });
});

test('a call that would freeze a value of this question is not cached', () => {
  const q = 'Kui suur on ruumi „TUALETTRUUM (24-3)“ korrusel „2.korrus“?';
  const { slots } = templatize(q);
  assert.equal(abstractArgs({ label: 'TUALETTRUUM', floor: '2.korrus' }, slots, q), null);
});

test('enum arguments are kept even when the word occurs in the question', () => {
  assert.deepEqual(abstractArgs({ type: 'door', subtype: 'single' }, [], 'How many doors?'), { type: 'door', subtype: 'single' });
});
