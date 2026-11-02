// A configurable GPT built on the tiny autograd engine, plus optimizers.
import { Tensor, add, attention, crossEntropy, embedding, gelu, layerNorm, linear, rows } from "./tensor";
import type { Optim, TrainConfig } from "./config";

export function rng(seed: number) { // mulberry32
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function gauss(r: () => number) { const u = Math.max(1e-12, r()), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

export interface Param { name: string; group: string; t: Tensor; decay: boolean }
export interface Forward {
  logits: Tensor; loss: Tensor | null;
  probs: (Float32Array | null)[]; // attention weights per layer [B, H, T, T]
  stream: Tensor[]; // residual stream after embedding and after each layer
}

export class GPT {
  params: Param[] = [];
  P: Record<string, Tensor> = {};
  constructor(public c: TrainConfig, public V = 256) {
    const r = rng(c.seed * 9973 + 17), D = c.dModel, M = 4 * D;
    const mat = (name: string, group: string, shape: number[]) => this.add(name, group, shape, true, () => 0.02 * gauss(r));
    const vec = (name: string, group: string, n: number, val: number) => this.add(name, group, [n], false, () => val);
    mat("tok", "embed", [V, D]);
    if (c.pos) mat("pos", "pos", [c.nCtx, D]);
    for (let l = 0; l < c.nLayer; l++) {
      const p = `b${l}.`, ga = `L${l + 1} attn`, gm = `L${l + 1} MLP`;
      if (c.attention) {
        if (c.layernorm) { vec(p + "ln1.g", ga, D, 1); vec(p + "ln1.b", ga, D, 0); }
        mat(p + "qkv.w", ga, [3 * D, D]); vec(p + "qkv.b", ga, 3 * D, 0);
        mat(p + "proj.w", ga, [D, D]); vec(p + "proj.b", ga, D, 0);
      }
      if (c.mlp) {
        if (c.layernorm) { vec(p + "ln2.g", gm, D, 1); vec(p + "ln2.b", gm, D, 0); }
        mat(p + "fc.w", gm, [M, D]); vec(p + "fc.b", gm, M, 0);
        mat(p + "out.w", gm, [D, M]); vec(p + "out.b", gm, D, 0);
      }
    }
    if (c.layernorm) { vec("lnf.g", "final LN", D, 1); vec("lnf.b", "final LN", D, 0); }
  }
  private add(name: string, group: string, shape: number[], decay: boolean, init: () => number) {
    const t = Tensor.zeros(shape, true); for (let i = 0; i < t.size; i++) t.data[i] = init();
    this.P[name] = t; this.params.push({ name, group, t, decay });
  }
  get nParams() { return this.params.reduce((a, p) => a + p.t.size, 0); }
  groups() { return [...new Set(this.params.map((p) => p.group))]; }

  /** ids/targets are B·T tokens laid out row-major. lastOnly: only compute logits for the last position of each row. */
  forward(ids: ArrayLike<number>, B: number, T: number, targets?: ArrayLike<number>, lastOnly = false): Forward {
    const c = this.c, P = this.P;
    let x = embedding(P.tok, ids);
    if (c.pos) { const pi = new Int32Array(B * T); for (let i = 0; i < B * T; i++) pi[i] = i % T; x = add(x, embedding(P.pos, pi)); }
    const stream = [x], probs: (Float32Array | null)[] = [];
    for (let l = 0; l < c.nLayer; l++) {
      const p = `b${l}.`;
      if (c.attention) {
        const h = c.layernorm ? layerNorm(x, P[p + "ln1.g"], P[p + "ln1.b"]) : x;
        const a = attention(linear(h, P[p + "qkv.w"], P[p + "qkv.b"]), B, T, c.nHead);
        const o = linear(a.out, P[p + "proj.w"], P[p + "proj.b"]);
        x = c.residual ? add(x, o) : o; probs.push(a.probs);
      } else probs.push(null);
      if (c.mlp) {
        const h = c.layernorm ? layerNorm(x, P[p + "ln2.g"], P[p + "ln2.b"]) : x;
        const o = linear(gelu(linear(h, P[p + "fc.w"], P[p + "fc.b"])), P[p + "out.w"], P[p + "out.b"]);
        x = c.residual ? add(x, o) : o;
      }
      stream.push(x);
    }
    if (lastOnly) { // pick the last position of the (single) row
      x = rows(x, B * T - 1, 1);
    }
    const h = c.layernorm ? layerNorm(x, P["lnf.g"], P["lnf.b"]) : x;
    const logits = linear(h, P.tok); // weight tying: unembedding = embeddingᵀ
    return { logits, loss: targets ? crossEntropy(logits, targets) : null, probs, stream };
  }
}

// ---------------------------------------------------------------- optimizers
export class Optimizer {
  m = new Map<Tensor, Float32Array>(); v = new Map<Tensor, Float32Array>(); t = 0;
  constructor(public kind: Optim, public wd = 0.05) {}
  step(params: Param[], lr: number) {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    for (const p of params) {
      const w = p.t.data, g = p.t.grad; if (!g) continue;
      if (this.kind === "sgd") { for (let i = 0; i < w.length; i++) w[i] -= lr * g[i]; continue; }
      let m = this.m.get(p.t); if (!m) { m = new Float32Array(w.length); this.m.set(p.t, m); }
      if (this.kind === "momentum") { for (let i = 0; i < w.length; i++) { m[i] = 0.9 * m[i] + g[i]; w[i] -= lr * m[i]; } continue; }
      let v = this.v.get(p.t); if (!v) { v = new Float32Array(w.length); this.v.set(p.t, v); }
      const decay = p.decay ? 1 - lr * this.wd : 1;
      for (let i = 0; i < w.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        w[i] = w[i] * decay - lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
      }
    }
  }
}
