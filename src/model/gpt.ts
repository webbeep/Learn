// TinyGPT forward pass in plain TypeScript that records *every* intermediate
// value, so the UI can show (and you can ask about) any number in the model.
import { Tokenizer } from "./tokenizer";

export interface Config { vocab_size: number; n_ctx: number; d_model: number; n_head: number; n_layer: number; d_mlp: number }
export interface Meta {
  config: Config; vocab: string[]; merges: [string, string][];
  tensors: { name: string; shape: number[]; offset: number }[];
  pca: [number, number][]; corpus_sample: string;
}
type Vec = Float32Array;
export interface Pred { id: number; p: number }

export interface LayerTrace {
  scores: number[][][]; // [head][query i][key j] scaled dot products (j<=i)
  attn: number[][][]; // softmaxed weights
  residPre: Vec[]; attnOut: Vec[]; residMid: Vec[];
  mlpHidden: Vec[]; mlpOut: Vec[]; residPost: Vec[];
  headOut: Vec[][]; // [head][pos] head contribution (pre-projection) for last token visual
  lensMid: Pred[]; lensPost: Pred[]; // "logit lens" at last position
}
export interface Trace {
  ids: number[];
  tokEmb: Vec[]; posEmb: Vec[]; x0: Vec[];
  lens0: Pred[];
  layers: LayerTrace[];
  logits: Vec; // last position
}

export class TinyGPT {
  cfg: Config; tok: Tokenizer; meta: Meta;
  W: Record<string, Float32Array> = {};
  constructor(meta: Meta, buf: ArrayBuffer) {
    this.meta = meta; this.cfg = meta.config;
    this.tok = new Tokenizer(meta.vocab, meta.merges);
    const all = new Float32Array(buf);
    for (const t of meta.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.W[t.name] = all.subarray(t.offset, t.offset + n);
    }
  }
  row(name: string, i: number, width = this.cfg.d_model) { return this.W[name].subarray(i * width, (i + 1) * width); }

  linear(x: Vec, name: string, nOut: number): Vec {
    const w = this.W[name + ".weight"], b = this.W[name + ".bias"], nIn = x.length;
    const y = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = b[o]; const off = o * nIn;
      for (let i = 0; i < nIn; i++) s += w[off + i] * x[i];
      y[o] = s;
    }
    return y;
  }
  layerNorm(x: Vec, name: string): Vec {
    const n = x.length; let mu = 0; for (const v of x) mu += v; mu /= n;
    let va = 0; for (const v of x) va += (v - mu) ** 2; va /= n;
    const inv = 1 / Math.sqrt(va + 1e-5), g = this.W[name + ".weight"], b = this.W[name + ".bias"];
    const y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = (x[i] - mu) * inv * g[i] + b[i];
    return y;
  }
  unembed(x: Vec): Vec {
    const h = this.layerNorm(x, "ln_f"), V = this.cfg.vocab_size, D = this.cfg.d_model, E = this.W["tok.weight"];
    const out = new Float32Array(V);
    for (let v = 0; v < V; v++) { let s = 0; for (let d = 0; d < D; d++) s += E[v * D + d] * h[d]; out[v] = s; }
    return out;
  }
  lens(x: Vec, k = 5): Pred[] { return topK(softmax(this.unembed(x)), k); }

  forward(idsIn: number[]): Trace {
    const { n_ctx, d_model: D, n_head: H, n_layer: L, d_mlp: M } = this.cfg;
    const ids = idsIn.slice(-n_ctx), T = ids.length, hd = D / H;
    const tokEmb = ids.map((id) => this.row("tok.weight", id));
    const posEmb = ids.map((_, t) => this.row("pos.weight", t));
    const x0 = tokEmb.map((e, t) => add(e, posEmb[t]));
    let x = x0;
    const layers: LayerTrace[] = [];
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`;
      const qkv = x.map((v) => this.linear(this.layerNorm(v, p + "ln1"), p + "attn_qkv", 3 * D));
      const scores: number[][][] = [], attn: number[][][] = [], headOut: Vec[][] = [];
      const concat = Array.from({ length: T }, () => new Float32Array(D));
      for (let h = 0; h < H; h++) {
        const sH: number[][] = [], aH: number[][] = [], oH: Vec[] = [];
        for (let i = 0; i < T; i++) {
          const s: number[] = [];
          for (let j = 0; j <= i; j++) {
            let d = 0;
            for (let c = 0; c < hd; c++) d += qkv[i][h * hd + c] * qkv[j][D + h * hd + c];
            s.push(d / Math.sqrt(hd));
          }
          const a = Array.from(softmax(Float32Array.from(s)));
          const o = new Float32Array(hd);
          for (let j = 0; j <= i; j++) for (let c = 0; c < hd; c++) o[c] += a[j] * qkv[j][2 * D + h * hd + c];
          concat[i].set(o, h * hd);
          sH.push(s); aH.push(a); oH.push(o);
        }
        scores.push(sH); attn.push(aH); headOut.push(oH);
      }
      const attnOut = concat.map((c) => this.linear(c, p + "attn_out", D));
      const residMid = x.map((v, t) => add(v, attnOut[t]));
      const mlpHidden = residMid.map((v) => this.linear(this.layerNorm(v, p + "ln2"), p + "mlp_in", M).map(gelu));
      const mlpOut = mlpHidden.map((h) => this.linear(h, p + "mlp_out", D));
      const residPost = residMid.map((v, t) => add(v, mlpOut[t]));
      layers.push({ scores, attn, headOut, residPre: x, attnOut, residMid, mlpHidden, mlpOut, residPost,
        lensMid: this.lens(residMid[T - 1]), lensPost: this.lens(residPost[T - 1]) });
      x = residPost;
    }
    return { ids, tokEmb, posEmb, x0, lens0: this.lens(x0[T - 1]), layers, logits: this.unembed(x[T - 1]) };
  }
}

export function add(a: Vec, b: Vec) { const y = new Float32Array(a.length); for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i]; return y; }
export function gelu(x: number) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))); }
export function softmax(x: ArrayLike<number>, temp = 1): Float32Array {
  let m = -Infinity; for (let i = 0; i < x.length; i++) m = Math.max(m, x[i] / temp);
  const y = new Float32Array(x.length); let s = 0;
  for (let i = 0; i < x.length; i++) { y[i] = Math.exp(x[i] / temp - m); s += y[i]; }
  for (let i = 0; i < x.length; i++) y[i] /= s;
  return y;
}
export function topK(p: ArrayLike<number>, k: number): Pred[] {
  return Array.from(p, (v, id) => ({ id, p: v })).sort((a, b) => b.p - a.p).slice(0, k);
}
export function norm(v: Vec) { let s = 0; for (const x of v) s += x * x; return Math.sqrt(s); }

/** Turn logits into a sampling distribution with temperature + top-k. */
export function distribution(logits: Vec, temperature: number, k: number): Pred[] {
  if (temperature <= 0.01) { const t = topK(logits, 1)[0]; return [{ id: t.id, p: 1 }]; }
  const p = softmax(logits, temperature);
  const top = topK(p, k), z = top.reduce((a, b) => a + b.p, 0);
  return top.map((t) => ({ id: t.id, p: t.p / z }));
}

let modelPromise: Promise<TinyGPT> | null = null;
export function loadModel() {
  modelPromise ??= Promise.all([
    fetch("/model/meta.json").then((r) => r.json()),
    fetch("/model/weights.bin").then((r) => r.arrayBuffer()),
  ]).then(([m, b]) => new TinyGPT(m, b));
  return modelPromise;
}
