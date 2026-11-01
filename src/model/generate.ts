import { distribution, norm, type Pred, type TinyGPT, type Trace } from "./gpt";

export interface Step { ctx: number[]; trace: Trace; dist: Pred[]; r: number; chosen: number }
export interface GenSettings { n: number; temperature: number; topK: number; seed: number }

function rng(seed: number) { // mulberry32 — reproducible "randomness"
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export function generate(m: TinyGPT, startIds: number[], s: GenSettings): Step[] {
  let ids = startIds.length ? startIds : m.tok.encode("once upon a time").map((t) => t.id);
  const rand = rng(s.seed), steps: Step[] = [];
  for (let i = 0; i < s.n; i++) {
    const ctx = ids.slice(-m.cfg.n_ctx);
    const trace = m.forward(ctx);
    const dist = distribution(trace.logits, s.temperature, s.topK);
    const r = rand(); let acc = 0, chosen = dist[dist.length - 1].id;
    for (const d of dist) { acc += d.p; if (r < acc) { chosen = d.id; break; } }
    steps.push({ ctx, trace, dist, r, chosen });
    ids = [...ids, chosen];
  }
  return steps;
}

/** Nearest neighbours of each vocab token by cosine similarity of embeddings. */
export function neighbours(m: TinyGPT, k = 4): number[][] {
  const V = m.cfg.vocab_size, rows = Array.from({ length: V }, (_, i) => m.row("tok.weight", i));
  const n = rows.map(norm);
  return rows.map((a, i) => rows.map((b, j) => {
    if (i === j) return { j, s: -2 };
    let s = 0; for (let d = 0; d < a.length; d++) s += a[d] * b[d];
    return { j, s: s / (n[i] * n[j] + 1e-9) };
  }).sort((x, y) => y.s - x.s).slice(0, k).map((x) => x.j));
}
