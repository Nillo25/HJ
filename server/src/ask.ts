/**
 * Разбор вопроса: кэш шаблонов -> модель -> функция -> ответ с доказательством.
 *
 * Путь дешёвый по построению: известный шаблон стоит $0, модель зовётся
 * только на новый вопрос, и результат тут же кладётся в кэш.
 */
import { z } from 'zod';
import { generateObject } from 'ai';
import { google } from '@ai-sdk/google';
import { cacheGet, cachePut, floorsOf, queryElements, type Element } from './db.js';
import { FNS, resolveDrawing } from './tools.js';

const MODEL = google('gemini-2.5-flash');

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

function evidenceOf(rows: Element[], drawingId: number, limit = 50) {
  const floors = new Map<number, string>(
    (floorsOf.all(drawingId) as any[]).map((f) => [f.id, f.name]),
  );
  return rows.slice(0, limit).map((r) => ({
    kind: r.source_ref && r.source_ref.length === 22 ? 'ifc' : 'geom',
    id: r.id, ref: r.source_ref, type: r.type, subtype: r.subtype,
    label: r.label, floor: r.floor_id ? floors.get(r.floor_id) ?? null : null,
    page: r.page, bbox: r.bbox,
  }));
}

/* ------------------------------------------------------------- главное */

export async function ask(drawingName: string, question: string) {
  const t0 = Date.now();
  const d = resolveDrawing(drawingName);
  if (!d) {
    return { status: 'refused', reason: 'чертёж не найден', missing: 'drawing',
             route: 'cache', cost_usd: 0, latency_ms: Date.now() - t0 };
  }

  const { template, slots } = templatize(question);
  let parsed: any = null;
  let route: 'cache' | 'llm' = 'cache';
  let cost = 0;

  const hit = cacheGet.get(template) as { fn: string; args: string } | undefined;
  if (hit) {
    parsed = { fn: hit.fn, ...fill(JSON.parse(hit.args), slots) };
  } else {
    route = 'llm';
    // контекст чертежа: без него «на третьем этаже» не превратится в «3.korrus»
    const floors = (floorsOf.all(d.id) as any[]).map((f) => f.name).filter(Boolean);
    const labels = [...new Set(
      (queryElements.all({ drawing_id: d.id, type: 'room', subtype: null,
                           floor: null, literal: null }) as Element[])
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
        'Если вопрос не ложится ни на одну функцию — выбирай unsupported.',
      prompt:
        `Этажи этого чертежа: ${floors.length ? floors.join(' | ') : 'нет'}\n` +
        `Подписи помещений: ${labels.join(' | ') || 'нет'}\n\n` +
        `Вопрос: ${question}`,
    });
    parsed = object;
    cost = estimateCost(usage);

    if (object.fn !== 'unsupported') {
      const { fn, ...args } = object as any;
      cachePut.run(template, fn, JSON.stringify(args));
    }
  }

  if (parsed.fn === 'unsupported') {
    return { status: 'refused', reason: parsed.reason ?? 'тип вопроса не поддержан',
             missing: 'template', route, cost_usd: cost,
             latency_ms: Date.now() - t0 };
  }

  const { fn, ...args } = parsed;
  const impl = (FNS as any)[fn];
  if (!impl) {
    return { status: 'refused', reason: `неизвестная функция ${fn}`,
             missing: 'template', route, cost_usd: cost,
             latency_ms: Date.now() - t0 };
  }

  const { value, rows } = impl({ drawing_id: d.id, ...args });

  if (value === null) {
    return { status: 'refused', reason: 'в данных не найдено', missing: 'element',
             searched: JSON.stringify(args), route, cost_usd: cost,
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
    evidence: evidenceOf(rows, d.id),
    cost_usd: cost,
    latency_ms: Date.now() - t0,
    fn,
    args,
  };
}

/** Грубая оценка; для панели стоимости хватает. */
function estimateCost(usage: any): number {
  const inTok = usage?.promptTokens ?? 0;
  const outTok = usage?.completionTokens ?? 0;
  return (inTok * 0.3 + outTok * 2.5) / 1e6;
}
