// Minimal shadcn-style primitives: Button, Card, Badge, Input, Table.
import type { ButtonHTMLAttributes, HTMLAttributes, InputHTMLAttributes, ReactNode, TdHTMLAttributes, ThHTMLAttributes } from 'react';

export const cn = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(' ');

export function Button({ className, variant = 'default', ...p }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'default' | 'outline' | 'ghost' }) {
  const v = {
    default: 'bg-slate-900 text-white hover:bg-slate-700 disabled:bg-slate-300',
    outline: 'border border-slate-300 bg-white hover:bg-slate-100 disabled:opacity-50',
    ghost: 'hover:bg-slate-100 disabled:opacity-50',
  }[variant];
  return (
    <button
      className={cn('inline-flex h-9 items-center justify-center rounded-md px-4 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:cursor-not-allowed', v, className)}
      {...p}
    />
  );
}

export const Input = ({ className, ...p }: InputHTMLAttributes<HTMLInputElement>) => (
  <input
    className={cn('h-9 w-full rounded-md border border-slate-300 bg-white px-3 text-sm placeholder:text-slate-400 focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-slate-900', className)}
    {...p}
  />
);

export const Card = ({ className, ...p }: HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('rounded-lg border border-slate-200 bg-white shadow-sm', className)} {...p} />
);
export const CardHeader = ({ className, ...p }: HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-3', className)} {...p} />
);
export const CardTitle = ({ className, ...p }: HTMLAttributes<HTMLHeadingElement>) => (
  <h3 className={cn('text-sm font-semibold text-slate-700', className)} {...p} />
);
export const CardBody = ({ className, ...p }: HTMLAttributes<HTMLDivElement>) => <div className={cn('p-4', className)} {...p} />;

const tones = {
  slate: 'bg-slate-100 text-slate-700',
  green: 'bg-emerald-100 text-emerald-800',
  amber: 'bg-amber-100 text-amber-900',
  red: 'bg-rose-100 text-rose-800',
  blue: 'bg-sky-100 text-sky-800',
};
export const Badge = ({ tone = 'slate', children, className }: { tone?: keyof typeof tones; children: ReactNode; className?: string }) => (
  <span className={cn('inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium', tones[tone], className)}>{children}</span>
);

export const Table = ({ className, ...p }: HTMLAttributes<HTMLTableElement>) => (
  <table className={cn('w-full border-collapse text-sm', className)} {...p} />
);
export const Th = ({ className, ...p }: ThHTMLAttributes<HTMLTableCellElement>) => (
  <th className={cn('sticky top-0 bg-slate-50 px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-slate-500', className)} {...p} />
);
export const Td = ({ className, ...p }: TdHTMLAttributes<HTMLTableCellElement>) => (
  <td className={cn('border-t border-slate-100 px-3 py-1.5', className)} {...p} />
);
