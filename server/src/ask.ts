/**
 * Разбор вопроса: кэш шаблонов -> модель -> функция -> ответ с доказательством.
 *
 * Путь дешёвый по построению: известный шаблон стоит $0, модель зовётся
 * только на новый вопрос, и результат тут же кладётся в кэш.
 */
import { z } from 'zod';
import { generateObject } from 'ai';
import { google } from '@ai-sdk/google';
import { createOpenAI } from '@ai-sdk/openai';
import { execFile } from 'node:child_process';
import path from 'node:path';
import { recordFailure, recordUsage, type Usage } from './usage.js';
import { cacheGet, cachePut, floorsOf, queryElements, type Session } from './db.js';
import { PYTHON } from './py.js';

/** Question parsing model. LLM_PROVIDER=openai|google, LLM_MODEL overrides the default. */
const PROVIDER = process.env.LLM_PROVIDER ?? (process.env.OPENAI_API_KEY ? 'openai' : 'google');
const MODEL_ID = process.env.LLM_MODEL ?? (PROVIDER === 'openai' ? 'gpt-4.1-mini' : 'gemini-2.5-flash');
const openai = createOpenAI({ baseURL: process.env.OPENAI_BASE_URL });   // unset = api.openai.com
const MODEL = PROVIDER === 'openai' ? openai(MODEL_ID) : google(MODEL_ID);
const HAS_KEY = PROVIDER === 'openai'
  ? !!process.env.OPENAI_API_KEY : !!process.env.GOOGLE_GENERATIVE_AI_API_KEY;

/**
 * Every model call goes through here: calls are spaced to stay under the provider's requests-per-minute
 * limit (LLM_RPM) and a rate-limit error is retried with a pause instead of becoming a wrong refusal.
 */
const MIN_GAP_MS = 60000 / Number(process.env.LLM_RPM ?? 40);
let nextSlot = 0;
async function gated<T>(fn: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    const now = Date.now();
    const at = Math.max(now, nextSlot);
    nextSlot = at + MIN_GAP_MS;
    if (at > now) await new Promise((r) => setTimeout(r, at - now));
    try {
      return await fn();
    } catch (e: any) {
      recordFailure(`${PROVIDER}:${MODEL_ID}`);
      const msg = String(e?.message ?? e) + String(e?.lastError?.message ?? '');
      if (attempt < 3 && /rate limit|429|rpm|too many/i.test(msg)) {
        nextSlot = Math.max(nextSlot, Date.now()) + 8000 * (attempt + 1);
        continue;
      }
      throw e;
    }
  }
}

/** Known question templates live in Python; a hit costs nothing. */
function pyTemplate(s: Session, question: string): Promise<any | null> {
  return new Promise((resolve) => {
    execFile(PYTHON, [
      path.resolve('../indexer/ask_cli.py'),
      '--db', s.dbPath, '--drawing', s.drawing.name, '--question', question,
    ], { timeout: 20000, maxBuffer: 8e6 }, (err, out) => {
      if (err) return resolve(null);
      try { resolve(JSON.parse(out)); } catch { resolve(null); }
    });
  });
}

/** Run a function chosen by the model through the Python code that answers the templates. */
function pyCall(s: Session, fn: string, args: Record<string, unknown>): Promise<any | null> {
  return new Promise((resolve) => {
    execFile(PYTHON, [
      path.resolve('../indexer/ask_cli.py'),
      '--db', s.dbPath, '--drawing', s.drawing.name, '--fn', fn, '--args', JSON.stringify(args),
    ], { timeout: 20000, maxBuffer: 16e6 }, (err, out) => {
      if (err) return resolve(null);
      try { resolve(JSON.parse(out)); } catch { resolve(null); }
    });
  });
}

/** Facts sheet of an IFC model, computed from the index (no model). */
function pyFacts(s: Session, question: string): Promise<any | null> {
  return new Promise((resolve) => {
    execFile(PYTHON, [
      path.resolve('../indexer/ifc_facts.py'),
      '--db', s.dbPath, '--drawing', s.drawing.name, '--q', question,
    ], { timeout: 20000, maxBuffer: 8e6 }, (err, out) => {
      if (err) return resolve(null);
      try { resolve(JSON.parse(out)); } catch { resolve(null); }
    });
  });
}

/** Free-form question about a whole model: the model only phrases facts computed in Python. */
async function report(d: Session, question: string, t0: number, prior: Usage[], priorCost: number, route: string) {
  const facts = await pyFacts(d, question);
  if (!facts || !Object.keys(facts).length) {
    return { status: 'refused', reason: 'no facts could be computed for this model', missing: 'data',
             route, cost_usd: priorCost, usage: prior, latency_ms: Date.now() - t0 };
  }
  const { object, usage } = await gated(() => generateObject({
    model: MODEL,
    temperature: 0,
    schema: z.object({
      enough: z.boolean().describe('true only if the facts contain what the question asks'),
      answer: z.string().describe('the answer as compact plain lines; if not enough, what is missing'),
    }),
    system:
      'You answer a question about one building model using ONLY the JSON facts given. ' +
      'Copy numbers exactly, never compute new ones, never invent. Answer in the language of the question, ' +
      'compactly, as plain lines. Set enough=false when the facts do not contain the answer. ' +
      'When a figure is marked as measured or estimated (a *_note field), say so and give it as an estimate. ' +
      'Percentages, counts and areas in the facts are already computed: use them, do not recompute.',
    prompt: `Facts:\n${JSON.stringify(facts)}\n\nQuestion: ${question}`,
  }));
  const cost = estimateCost(usage);
  const used = [...prior, recordUsage(`${PROVIDER}:${MODEL_ID}`, usage?.promptTokens ?? 0, usage?.completionTokens ?? 0, cost)];
  if (!object.enough) {
    // never present "I don't know" as an answer
    return { status: 'refused', reason: object.answer.trim(), missing: 'data', route,
             cost_usd: priorCost + cost, usage: used, latency_ms: Date.now() - t0, fn: 'report', args: {} };
  }
  return {
    status: 'answered', answer: object.answer.trim(), unit: null, confidence: 1, route,
    evidence: [], cost_usd: priorCost + cost, usage: used, latency_ms: Date.now() - t0,
    fn: 'report', args: {},
  };
}

/** Lines of a PDF's text that the question is about (chosen in Python: no model, no tokens). */
function pySnippets(s: Session, question: string): Promise<{ id: number; page: number; text: string; bbox: number[]; x?: number; y?: number }[]> {
  return new Promise((resolve) => {
    execFile(PYTHON, [
      path.resolve('../indexer/ask_cli.py'),
      '--db', s.dbPath, '--drawing', s.drawing.name, '--snippets', '--question', question,
    ], { timeout: 20000, maxBuffer: 16e6 }, (err, out) => {
      if (err) return resolve([]);
      try { resolve(JSON.parse(out)); } catch { resolve([]); }
    });
  });
}

/**
 * A question about the text of a PDF: the model reads only the chosen numbered lines and must name the ones it used.
 * Those lines become the evidence and are boxed on the page; an answer that cites nothing is not shown as an answer.
 */
async function textAnswer(d: Session, question: string, t0: number) {
  if (!HAS_KEY) {
    return { status: 'refused', reason: 'free-form questions about a PDF need a model key', missing: 'template',
             route: 'none', usage: [], cost_usd: 0, latency_ms: Date.now() - t0 };
  }
  const snips = await pySnippets(d, question);
  if (!snips.length) {
    return { status: 'refused', reason: 'No line of text in this PDF relates to the question (it may be a scan or a drawing without text).',
             missing: 'data', route: 'cache', usage: [], cost_usd: 0, latency_ms: Date.now() - t0 };
  }
  const byId = new Map(snips.map((x) => [x.id, x]));
  const { object, usage } = await gated(() => generateObject({
    model: MODEL,
    temperature: 0,
    schema: z.object({
      enough: z.boolean().describe('true only if the numbered lines contain the answer'),
      answer: z.string().describe('the answer, short, in the language of the question; if not enough: what is missing'),
      refs: z.array(z.number()).describe('ids of the lines the answer is taken from (at most 40)'),
    }),
    system:
      'You answer a question about a construction drawing using ONLY the numbered lines of text extracted from its PDF. ' +
      'Copy values exactly. In a table, read a value under the header at the same x, not a neighbouring column. Count only items that are explicitly listed in the lines; if the list may be incomplete or the ' +
      'lines do not contain the answer, set enough=false. Put the ids of the lines you used in refs. Never invent an id.',
    prompt: `Lines (#id pPAGE xCOLUMN yROW: text). Lines of a table row share the same y; a value belongs to the column header whose x is closest:\n${snips.map((x) => `#${x.id} p${x.page} x${x.x ?? 0} y${x.y ?? 0}: ${x.text}`).join('\n')}\n\nQuestion: ${question}`,
  }));
  const cost = estimateCost(usage);
  const used = [recordUsage(`${PROVIDER}:${MODEL_ID}`, usage?.promptTokens ?? 0, usage?.completionTokens ?? 0, cost)];
  const refs = [...new Set(object.refs)].filter((r) => byId.has(r)).slice(0, 40);
  if (!object.enough) {
    return { status: 'refused', reason: object.answer.trim(), missing: 'data', route: 'llm', cost_usd: cost, usage: used,
             latency_ms: Date.now() - t0, fn: 'text_search', args: {} };
  }
  if (!refs.length) {
    return { status: 'refused', reason: 'The answer did not point to any line of the PDF, so it cannot be shown as verified.',
             missing: 'data', route: 'llm', cost_usd: cost, usage: used, latency_ms: Date.now() - t0, fn: 'text_search', args: {} };
  }
  return {
    status: 'answered', answer: object.answer.trim(), unit: 'text', confidence: 1, route: 'llm', cost_usd: cost, usage: used,
    latency_ms: Date.now() - t0, fn: 'text_search', args: {},
    evidence: refs.map((r) => {
      const x = byId.get(r)!;
      return { kind: 'page', id: x.id, label: x.text, page: x.page, bbox: JSON.stringify(x.bbox) };
    }),
  };
}

/* ------------------------------------------------ схема разбора вопроса */

const Query = z.discriminatedUnion('fn', [
  z.object({
    fn: z.literal('count_objects'),
    type: z.enum(['room', 'door', 'window', 'fixture', 'apartment', 'wall',
                  'column', 'beam', 'stair', 'slab']),
    subtype: z.string().optional()
      .describe('type=fixture: toilet (WC bowls, "how many WC/toilets drawn"), sink (washbasins), bathtub, shower. ' +
                'type=room: bedroom, kitchen, bathroom, wc (a separate WC ROOM), sauna, balcony. ' +
                'type=door: single, double, sliding.'),
    literal: z.string().optional()
      .describe('ТОЛЬКО если в вопросе подпись взята в кавычки, например ' +
                "rooms labelled 'Cucina' -> literal: Cucina. Иначе не заполнять."),
    floor: z.string().optional().describe('точное имя этажа из списка ниже'),
  }),
  z.object({
    fn: z.literal('room_area'),
    label: z.string().describe(
      'ONE room. If the question gives a room NUMBER (0.22, 5.03, U1.04) the label is that number and the name in ' +
      'brackets is ignored. If it names a room with its number in brackets, e.g. "TUALETTRUUM (24-3)", copy all of it.'),
    floor: z.string().optional(),
  }),
  z.object({
    fn: z.literal('sum_quantity'),
    type: z.string(),
    field: z.enum(['area', 'length', 'volume']),
    subtype: z.string().optional(),
    floor: z.string().optional(),
    only_numbered: z.boolean().optional()
      .describe('true для «всех пронумерованных внутренних помещений»'),
    label: z.string().optional()
      .describe('only for type=room: sum over the rooms with EXACTLY this printed label, e.g. "all rooms labelled \'Zimmer\'"'),
  }),
  z.object({
    fn: z.literal('apartment_area'),
    label: z.string().describe('the apartment letter as written on the plan: A, B, C...'),
  }),
  z.object({
    fn: z.literal('floor_area'),
    floor: z.string().optional(),
  }).describe('TOTAL area of the whole floor / storey / plan, with or without parts (apartments, kinds of room). ' +
              'Use it for "общая площадь этажа", "total area of the floor", "how big is the plan". Not for one room or one kind of room.'),
  z.object({ fn: z.literal('list_floors') }),
  z.object({ fn: z.literal('report') }).describe('free-form question about a whole IFC model: software, levels, space breakdown, summary'),
  z.object({
    fn: z.literal('unsupported'),
    reason: z.string().describe('one short sentence IN ENGLISH: why no function can answer this question'),
  }),
]);

/* ------------------------------------------------------- кэш шаблонов */

/** «Сколько окон на плане "3.korrus"?» -> «сколько окон на плане {0}» + ["3.korrus"] */
/** Exported for the tests. */
export function templatize(q: string): { template: string; slots: string[] } {
  const slots: string[] = [];
  // one pass: a quoted label or a number becomes the next slot; digits inside a quoted label stay in it
  let t = q.trim().toLowerCase().replace(
    /['"\u201c\u201d\u201e\u00ab\u00bb]([^'"\u201c\u201d\u201e\u00ab\u00bb]+)['"\u201c\u201d\u201e\u00ab\u00bb]|\b\d+(?:[.,]\d+)*\b/g,
    (m: string, quoted?: string) => `{${slots.push(quoted ?? m) - 1}}`,
  );
  t = t.replace(/\s+/g, ' ').replace(/[?!.]+$/, '');
  return { template: t, slots };
}

/**
 * Turn the arguments of a parsed question into a reusable template: a value copied from a quoted
 * label or a number becomes {i}. A value copied from the question that is NOT a slot would be frozen
 * into the template and answer every similar question with it, so such a result is not cached.
 */
const FREE_TEXT = new Set(['label', 'floor', 'literal']);
/** Exported for the tests. */
export function abstractArgs(args: Record<string, unknown>, slots: string[], question: string): Record<string, unknown> | null {
  const q = question.toLowerCase();
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(args)) {
    if (typeof v !== 'string') { out[k] = v; continue; }
    const low = v.toLowerCase();
    const exact = slots.findIndex((x) => x && x.toLowerCase() === low);
    if (exact >= 0) { out[k] = `{${exact}}`; continue; }
    let w = v, hit = false;
    slots.forEach((x, i) => {
      if (x && w.toLowerCase().includes(x.toLowerCase())) {
        w = w.replace(new RegExp(x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'ig'), `{${i}}`);
        hit = true;
      }
    });
    if (hit) { out[k] = w; continue; }
    // only free-text arguments can freeze a value of this question into the template; type, subtype and field are enums
    if (FREE_TEXT.has(k) && low.length > 2 && q.includes(low)) return null;
    out[k] = v;
  }
  return out;
}

function fill(args: any, slots: string[]) {
  const out: any = {};
  for (const [k, v] of Object.entries(args)) {
    out[k] = typeof v === 'string'
      ? v.replace(/\{(\d+)\}/g, (_, i) => slots[Number(i)] ?? '')
      : v;
  }
  return out;
}

/* ------------------------------------------------------------- главное */

export async function ask(d: Session, question: string) {
  const t0 = Date.now();
  const drawingId = d.drawing.id;

  const tpl = await pyTemplate(d, question);
  // a PDF without schedule rows has no "parsed objects" but has text: it falls through to the text path
  const fallsThrough = tpl?.status === 'refused' && (tpl.missing === 'template' || (d.meta.kind === 'pdf' && (tpl.missing === 'parser' || tpl.missing === 'source_model')));
  if (tpl && !fallsThrough) {
    return { ...tpl, route: 'template', usage: [], latency_ms: Date.now() - t0 };
  }

  // a PDF is read, not queried: the question is answered from the lines of its text
  if (d.meta.kind === 'pdf') return textAnswer(d, question, t0);

  const { template, slots } = templatize(question);
  let parsed: any = null;
  let route: 'cache' | 'llm' = 'cache';
  let cost = 0;
  const used: Usage[] = [];

  const hit = cacheGet(template);
  if (hit) {
    parsed = { fn: hit.fn, ...fill(JSON.parse(hit.args), slots) };
  } else {
    route = 'llm';
    if (!HAS_KEY) {
      return { status: 'refused', reason: 'question is not a known template and no model key is configured',
               missing: 'template', route: 'none', usage: [], cost_usd: 0, latency_ms: Date.now() - t0 };
    }
    // контекст чертежа: без него «на третьем этаже» не превратится в «3.korrus»
    const floors = floorsOf(d.db, drawingId).map((f) => f.name).filter(Boolean);
    const labels = [...new Set(
      queryElements(d.db, { drawing_id: drawingId, type: 'room', subtype: null,
                            floor: null, literal: null })
        .map((r) => r.label).filter(Boolean),
    )].slice(0, 120);

    const { object, usage } = await gated(() => generateObject({
      model: MODEL,
      schema: Query,
      temperature: 0,
      system:
        'Ты превращаешь вопрос о строительном чертеже в вызов функции. ' +
        'Отвечай только выбором функции и аргументов, ничего не вычисляй. ' +
        'Вопрос может быть на любом языке. Поле reason (если нужно) пиши по-английски. ' +
        'Если вопрос не ложится ни на одну функцию — выбирай unsupported. ' +
        'Если вопрос просит сумму, разность или сочетание РАЗНЫХ типов объектов («стены и колонны вместе»), ' +
        'одной функцией его не посчитать: выбирай unsupported, не отвечай только на часть. ' +
        'Название помещения копируй из вопроса целиком, вместе с номером в скобках. ' +
        'Общая площадь всего этажа/плана без указания типа помещения ("общая площадь этажа", "total area of the floor") — это floor_area, не sum_quantity. ' +
        (d.meta.kind === 'ifc'
          ? 'Для IFC-модели описательные вопросы (ПО и версия, перечень, группировка, распределение, ' +
            'самое большое/узкое/высокое, сводка по уровням) — это report, а не unsupported. '
          : ''),
      prompt:
        `Этажи этого чертежа: ${floors.length ? floors.join(' | ') : 'нет'}\n` +
        `Подписи помещений: ${labels.join(' | ') || 'нет'}\n\n` +
        `Вопрос: ${question}`,
    }));
    parsed = object;
    cost = estimateCost(usage);
    used.push(recordUsage(`${PROVIDER}:${MODEL_ID}`, usage?.promptTokens ?? 0,
                          usage?.completionTokens ?? 0, cost));

    if (object.fn !== 'unsupported' && object.fn !== 'report') {
      const { fn, ...args } = object as any;
      // the cache keeps a template, not an answer: values taken from the question must become slots
      const abstract = abstractArgs(args, slots, question);
      if (abstract) cachePut(template, fn, JSON.stringify(abstract));
    }
  }

  // a descriptive IFC question that no function covers: phrase it from the facts sheet
  if (parsed.fn === 'unsupported' && d.meta.kind === 'ifc') return report(d, question, t0, used, cost, route);

  if (parsed.fn === 'unsupported') {
    return { status: 'refused', reason: parsed.reason ?? 'this kind of question is not supported',
             missing: 'template', route, cost_usd: cost, usage: used,
             latency_ms: Date.now() - t0 };
  }

  if (parsed.fn === 'report') return report(d, question, t0, used, cost, route);

  const { fn, ...args } = parsed;
  const bad = invalidPair(args);
  if (bad) {
    return { status: 'refused', reason: bad, missing: 'template', route, cost_usd: cost, usage: used,
             latency_ms: Date.now() - t0 };
  }
  // the function runs in Python, the same code the templates use: one source of truth for the numbers
  const res = await pyCall(d, fn, args);
  if (!res) {
    return { status: 'refused', reason: 'the indexer could not run this call', missing: 'internal',
             route, cost_usd: cost, usage: used, latency_ms: Date.now() - t0 };
  }
  return { ...res, route, cost_usd: cost, usage: used, latency_ms: Date.now() - t0 };
}

/** The model may combine a type with a subtype that cannot exist: answering 0 for it would be a wrong answer. */
const SUBTYPES: Record<string, string[]> = {
  fixture: ['toilet', 'sink', 'bathtub', 'shower', 'other'],
  door: ['single', 'double', 'sliding', 'opening'],
  room: ['bedroom', 'kitchen', 'bathroom', 'wc', 'sauna', 'balcony', 'corridor', 'living', 'storage', 'technical', 'other'],
};
function invalidPair(a: any): string | null {
  if (!a.subtype || !a.type) return null;
  const ok = SUBTYPES[a.type];
  if (!ok) return `no subtypes are defined for ${a.type}`;
  if (a.type === 'fixture' && a.subtype === 'wc') { a.subtype = 'toilet'; return null; }   // the one unambiguous synonym
  return ok.includes(a.subtype) ? null : `"${a.subtype}" is not a known kind of ${a.type}`;
}

/** Грубая оценка; для панели стоимости хватает. */
function estimateCost(usage: any): number {
  const inTok = usage?.promptTokens ?? 0;
  const outTok = usage?.completionTokens ?? 0;
  const pin = Number(process.env.LLM_PRICE_IN ?? 0.3);    // USD per 1M tokens
  const pout = Number(process.env.LLM_PRICE_OUT ?? 2.5);
  return (inTok * pin + outTok * pout) / 1e6;
}
