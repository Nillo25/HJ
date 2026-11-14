/**
 * Функции поверх базы. Их же вызывает модель через tools.
 *
 * Правило: каждая возвращает И число, И элементы. Функция, вернувшая
 * число без элементов, — ошибка: из элементов собирается evidence,
 * а без доказательства ответ не засчитывается (25% оценки).
 */
import { queryElements, floorsOf, drawingByName, type Element } from './db.js';

export type ToolResult = { value: number | null; rows: Element[] };

type Base = { drawing_id: number };

export function count_objects(
  a: Base & { type: string; subtype?: string; floor?: string; literal?: string },
): ToolResult {
  const rows = queryElements.all({
    drawing_id: a.drawing_id,
    type: a.type,
    subtype: a.subtype ?? null,
    floor: a.floor ?? null,
    literal: a.literal ?? null,
  }) as Element[];
  return { value: rows.length, rows };
}

export function room_area(
  a: Base & { label: string; floor?: string },
): ToolResult {
  const all = queryElements.all({
    drawing_id: a.drawing_id,
    type: 'room',
    subtype: null,
    floor: a.floor ?? null,
    literal: a.label,
  }) as Element[];
  if (!all.length) return { value: null, rows: [] };
  return { value: all.reduce((s, r) => s + (r.area ?? 0), 0), rows: all };
}

export function sum_quantity(
  a: Base & {
    type: string;
    field: 'area' | 'length' | 'volume';
    subtype?: string;
    floor?: string;
    only_numbered?: boolean;
  },
): ToolResult {
  let rows = queryElements.all({
    drawing_id: a.drawing_id,
    type: a.type,
    subtype: a.subtype ?? null,
    floor: a.floor ?? null,
    literal: null,
  }) as Element[];

  if (a.only_numbered) {
    // «все пронумерованные внутренние помещения, кроме балконов и шахт»
    rows = rows.filter((r) => {
      try {
        return JSON.parse(r.props ?? '{}').numbered_indoor === true;
      } catch {
        return false;
      }
    });
  }
  if (!rows.length) return { value: null, rows: [] };
  return { value: rows.reduce((s, r) => s + ((r[a.field] as number) ?? 0), 0), rows };
}

export function list_floors(a: Base): ToolResult {
  const rows = floorsOf.all(a.drawing_id) as unknown as Element[];
  return { value: rows.length, rows };
}

export function resolveDrawing(name: string): { id: number; row: any } | null {
  const row = drawingByName.get(name) as any;
  return row ? { id: row.id, row } : null;
}

export const FNS = { count_objects, room_area, sum_quantity, list_floors };
export type FnName = keyof typeof FNS;
