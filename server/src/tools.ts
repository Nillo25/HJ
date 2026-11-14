/**
 * Functions over a session database. The model calls them through the parsed question.
 *
 * Rule: each returns BOTH the number and the elements. A function that returns
 * a number without elements is a bug: the evidence is built from the elements.
 */
import type Database from 'better-sqlite3';
import { queryElements, floorsOf, type Element } from './db.js';

export type ToolResult = { value: number | null; rows: Element[] };

type Base = { db: Database.Database; drawing_id: number };

export function count_objects(
  a: Base & { type: string; subtype?: string; floor?: string; literal?: string },
): ToolResult {
  const rows = queryElements(a.db, {
    drawing_id: a.drawing_id, type: a.type, subtype: a.subtype ?? null,
    floor: a.floor ?? null, literal: a.literal ?? null,
  });
  return { value: rows.length, rows };
}

export function room_area(a: Base & { label: string; floor?: string }): ToolResult {
  const all = queryElements(a.db, {
    drawing_id: a.drawing_id, type: 'room', subtype: null,
    floor: a.floor ?? null, literal: a.label,
  });
  if (!all.length) return { value: null, rows: [] };
  return { value: all.reduce((s, r) => s + (r.area ?? 0), 0), rows: all };
}

export function sum_quantity(
  a: Base & {
    type: string; field: 'area' | 'length' | 'volume';
    subtype?: string; floor?: string; only_numbered?: boolean;
  },
): ToolResult {
  let rows = queryElements(a.db, {
    drawing_id: a.drawing_id, type: a.type, subtype: a.subtype ?? null,
    floor: a.floor ?? null, literal: null,
  });
  if (a.only_numbered) {
    // all numbered indoor rooms, balconies and shafts excluded
    rows = rows.filter((r) => {
      try { return JSON.parse(r.props ?? '{}').numbered_indoor === true; } catch { return false; }
    });
  }
  if (!rows.length) return { value: null, rows: [] };
  return { value: rows.reduce((s, r) => s + ((r[a.field] as number) ?? 0), 0), rows };
}

export function list_floors(a: Base): ToolResult {
  const rows = floorsOf(a.db, a.drawing_id) as unknown as Element[];
  return { value: rows.length, rows };
}

export const FNS = { count_objects, room_area, sum_quantity, list_floors };
export type FnName = keyof typeof FNS;
