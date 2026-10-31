export const pct = (v: number | null | undefined, digits = 0) => (v == null ? 'n/a' : `${(100 * v).toFixed(digits)}%`);
export const num = (v: number | null | undefined, digits = 0) =>
  v == null ? 'n/a' : v.toLocaleString('en-GB', { maximumFractionDigits: digits, minimumFractionDigits: digits });
export const day = (iso: string | null | undefined) => (iso ? iso.slice(0, 10) : 'not recorded');
export const human = (s: string) => s.replaceAll('_', ' ');
