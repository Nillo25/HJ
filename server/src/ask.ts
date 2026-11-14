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
import { recordUsage, type Usage } from './usage.js';
import { cacheGet, cachePut, floorsOf, queryElements, type Element, type Session } from './db.js';
import { FNS } from './tools.js';
import { PYTHON } from './py.js';

/** Question parsing model. LLM_PROVIDER=openai|google, LLM_MODEL overrides the default. */
const PROVIDER = process.env.LLM_PROVIDER ?? (process.env.OPENAI_API_KEY ? 'openai' : 'google');
const MODEL_ID = process.env.LLM_MODEL ?? (PROVIDER === 'openai' ? 'gpt-4.1-mini' : 'gemini-2.5-flash');
const openai = createOpenAI({ baseURL: process.env.OPENAI_BASE_URL });   // unset = api.openai.com
const MODEL = PROVIDER === 'openai' ? openai(MODEL_ID) : google(MODEL_ID);
const HAS_KEY = PROVIDER === 'openai'
  ? !!process.env.OPENAI_API_KEY : !!process.env.GOOGLE_GENERATIVE_AI_API_KEY;

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

/** Facts sheet of an IFC model, computed from the index (no model). */
function pyFacts(s: Session): Promise<any | null> {
  return new Promise((resolve) => {
    execFile(PYTHON, [
      path.resolve('../indexer/ifc_facts.py'),
      '--db', s.dbPath, '--drawing', s.drawing.name,
    ], { timeout: 20000, maxBuffer: 8e6 }, (err, out) => {
      if (err) return resolve(null);
      try { resolve(JSON.parse(out)); } catch { resolve(null); }
    });
  });
}

/** Free-form question about a whole model: the model only phrases facts computed in Python. */
async function report(d: Session, question: string, t0: number, prior: Usage[], priorCost: number, route: string) {
  const facts = await pyFacts(d);
  if (!facts || !Object.keys(facts).length) {
    return { status: 'refused', reason: 'no facts could be computed for this model', missing: 'data',
             route, cost_usd: priorCost, usage: prior, latency_ms: Date.now() - t0 };
  }
  const { object, usage } = await generateObject({
    model: MODEL,
    temperature: 0,
    schema: z.object({
      enough: z.boolean().describe('true only if the facts contain what the question asks'),
      answer: z.string().describe('the answer as compact plain lines; if not enough, what is missing'),
    }),
    system:
      'You answer a question about one building model using ONLY the JSON facts given. ' +
      'Copy numbers exactly, never compute new ones, never invent. Answer in the language of the question, ' +
      'compactly, as plain lines. Set enough=false when the facts do not contain the answer.',
    prompt: `Facts:\n${JSON.stringify(facts)}\n\nQuestion: ${question}`,
  });
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

/* ------------------------------------------------ схема разбора вопроса */

const Query = z.discriminatedUnion('fn', [
  z.object({
    fn: z.literal('count_objects'),
    type: z.enum(['room', 'door', 'window', 'fixture', 'apartment', 'wall',
                  'column', 'beam', 'stair', 'slab']),
    subtype: z.string().optional()
      .describe('смысловой тип: bedroom, kitchen, bathroom, wc, sauna, balcony, ' +
                'toilet, sink, bathtub, shower'),
    literal: z.string().optional()
      .describe('ТОЛЬКО если в вопросе подпись взята в кавычки, например ' +
                "rooms labelled 'Cucina' -> literal: Cucina. Иначе не заполнять."),
    floor: z.string().optional().describe('точное имя этажа из списка ниже'),
  }),
  z.object({
    fn: z.literal('room_area'),
    label: z.string(),
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
  }),
  z.object({ fn: z.literal('list_floors') }),
  z.object({ fn: z.literal('report') }).describe('free-form question about a whole IFC model: software, levels, space breakdown, summary'),
  z.object({
    fn: z.literal('unsupported'),
    reason: z.string().describe('почему вопрос не ложится ни на одну функцию'),
  }),
]);

/* ------------------------------------------------------- кэш шаблонов */

/** «Сколько окон на плане "3.korrus"?» -> «сколько окон на плане {0}» + ["3.korrus"] */
export function templatize(q: string): { template: string; slots: string[] } {
  const slots: string[] = [];
  let t = q.trim().toLowerCase();
  t = t.replace(/['"“”„«»]([^'"“”„«»]+)['"“”„«»]/g, () => `{${slots.push(RegExp.$1) - 1}}`);
  t = t.replace(/\b\d+([.,]\d+)*\b/g, () => `{${slots.push(RegExp.lastMatch) - 1}}`);
  t = t.replace(/\s+/g, ' ').replace(/[?!.]+$/, '');
  return { template: t, slots };
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

/* ------------------------------------------------------- доказательство */

function scheduleLabel(r: Element): string | null {
  if (r.type !== 'room_row' && r.type !== 'door_row') return r.label;
  try {
    const p = JSON.parse(r.props ?? '{}');
    const name = p.name ?? p.room_name;
    return name ? `${r.label} ${name}` : r.label;
  } catch { return r.label; }
}

function evidenceOf(d: Session, rows: Element[], limit = 1000) {
  const floors = new Map<number, string>(
    floorsOf(d.db, d.drawing.id).map((f) => [f.id, f.name]),
  );
  return rows.slice(0, limit).map((r) => ({
    kind: r.source_ref && r.source_ref.length === 22 ? 'ifc' : 'geom',
    id: r.id, ref: r.source_ref, type: r.type, subtype: r.subtype,
    label: scheduleLabel(r), floor: r.floor_id ? floors.get(r.floor_id) ?? null : null, floor_id: r.floor_id,
    page: r.page, bbox: r.bbox, area: r.area, length: r.length, volume: r.volume,
  }));
}

/* ------------------------------------------------------------- главное */

export async function ask(d: Session, question: string) {
  const t0 = Date.now();
  const drawingId = d.drawing.id;

  const tpl = await pyTemplate(d, question);
  if (tpl && !(tpl.status === 'refused' && tpl.missing === 'template')) {
    return { ...tpl, route: 'template', usage: [], latency_ms: Date.now() - t0 };
  }

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

    const { object, usage } = await generateObject({
      model: MODEL,
      schema: Query,
      temperature: 0,
      system:
        'Ты превращаешь вопрос о строительном чертеже в вызов функции. ' +
        'Отвечай только выбором функции и аргументов, ничего не вычисляй. ' +
        'Вопрос может быть на любом языке. ' +
        'Если вопрос не ложится ни на одну функцию — выбирай unsupported. ' +
        (d.meta.kind === 'ifc'
          ? 'Для IFC-модели описательные вопросы (ПО и версия, перечень, группировка, распределение, ' +
            'самое большое/узкое/высокое, сводка по уровням) — это report, а не unsupported. '
          : ''),
      prompt:
        `Этажи этого чертежа: ${floors.length ? floors.join(' | ') : 'нет'}\n` +
        `Подписи помещений: ${labels.join(' | ') || 'нет'}\n\n` +
        `Вопрос: ${question}`,
    });
    parsed = object;
    cost = estimateCost(usage);
    used.push(recordUsage(`${PROVIDER}:${MODEL_ID}`, usage?.promptTokens ?? 0,
                          usage?.completionTokens ?? 0, cost));

    if (object.fn !== 'unsupported') {
      const { fn, ...args } = object as any;
      cachePut(template, fn, JSON.stringify(args));
    }
  }

  // a descriptive IFC question that no function covers: phrase it from the facts sheet
  if (parsed.fn === 'unsupported' && d.meta.kind === 'ifc') return report(d, question, t0, used, cost, route);

  if (parsed.fn === 'unsupported') {
    return { status: 'refused', reason: parsed.reason ?? 'тип вопроса не поддержан',
             missing: 'template', route, cost_usd: cost, usage: used,
             latency_ms: Date.now() - t0 };
  }

  if (parsed.fn === 'report') return report(d, question, t0, used, cost, route);

  const { fn, ...args } = parsed;
  const impl = (FNS as any)[fn];
  if (!impl) {
    return { status: 'refused', reason: `неизвестная функция ${fn}`,
             missing: 'template', route, cost_usd: cost, usage: used,
             latency_ms: Date.now() - t0 };
  }

  const { value, rows } = impl({ db: d.db, drawing_id: drawingId, ...args });

  if (value === null) {
    return { status: 'refused', reason: 'в данных не найдено', missing: 'element',
             searched: JSON.stringify(args), route, cost_usd: cost, usage: used,
             latency_ms: Date.now() - t0 };
  }

  const unit = fn === 'room_area' || args.field === 'area' ? 'm2' : 'count';
  const confidence = rows.length
    ? Math.min(...rows.map((r: Element) => r.confidence ?? 1))
    : 1;

  return {
    status: confidence >= 1 ? 'answered' : 'estimated',
    answer: unit === 'm2' ? Math.round(value * 100) / 100 : value,
    unit,
    confidence,
    route,
    evidence: evidenceOf(d, rows),
    cost_usd: cost,
    usage: used,
    latency_ms: Date.now() - t0,
    fn,
    args,
  };
}

/** Грубая оценка; для панели стоимости хватает. */
function estimateCost(usage: any): number {
  const inTok = usage?.promptTokens ?? 0;
  const outTok = usage?.completionTokens ?? 0;
  const pin = Number(process.env.LLM_PRICE_IN ?? 0.3);    // USD per 1M tokens
  const pout = Number(process.env.LLM_PRICE_OUT ?? 2.5);
  return (inTok * pin + outTok * pout) / 1e6;
}
