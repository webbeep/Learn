// Shared visual helpers.
import { show } from "../model/tokenizer";
export { show };

export const tokColor = (id: number, l = 62) => `hsl(${(id * 137.508) % 360} 70% ${l}%)`;

/** Diverging color for a signed value, blue (−) → dark (0) → orange (+). */
export function div(v: number, scale = 1) {
  const t = Math.max(-1, Math.min(1, v / scale));
  const base = [28, 33, 48], pos = [255, 150, 60], neg = [80, 150, 255];
  const c = t >= 0 ? pos : neg, a = Math.abs(t) ** 0.8;
  return `rgb(${base.map((b, i) => Math.round(b + (c[i] - b) * a)).join(",")})`;
}
/** Sequential color 0..1 dark → bright yellow. */
export function seq(v: number) {
  const t = Math.max(0, Math.min(1, v)) ** 0.6;
  const a = [24, 28, 44], b = [255, 214, 90];
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(",")})`;
}
export const pct = (p: number) => (p >= 0.995 ? "100%" : p < 0.001 ? "<0.1%" : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`);
export const f2 = (x: number) => (x >= 0 ? " " : "") + x.toFixed(2);
export const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
export const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
/** Map scene progress t∈[0,1] to a local stage progress between a and b. */
export const phase = (t: number, a: number, b: number) => ease(clamp((t - a) / (b - a), 0, 1));
