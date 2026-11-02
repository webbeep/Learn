// Everything a training run needs, independent of the Web Worker (so it can be benchmarked in Node).
import { Tokenizer } from "../model/tokenizer";
import { backward, l2, noGrad, recording } from "./tensor";
import { GPT, Optimizer, rng } from "./model";
import { PROBE, PROMPTS, type TrainConfig } from "./config";

export interface Viz {
  step: number;
  groups: { name: string; grad: number; weight: number; update: number }[]; // per parameter group: ‖grad‖, ‖W‖, ‖ΔW‖/‖W‖
  stream: { name: string; grad: number }[]; // ‖d loss / d residual stream‖ at each depth
  attn: { tokens: string[]; layers: (number[][][] | null)[] }; // [layer][head][i][j] on the probe prompt
  probe: { token: string; p: number }[]; pRed: number;
  hist: { lo: number; hi: number; now: number[]; init: number[]; std: number; initStd: number };
}
export interface Sample { prompt: string; text: string }

const WARMUP = 30;

export class Trainer {
  model: GPT; opt: Optimizer; step = 0; tokensSeen = 0; lastLoss = NaN; diverged = false;
  private r: () => number; private evalIdx: number[];
  private initW: Float32Array;
  private prevW = new Map<string, Float32Array>(); private lastGrads: { name: string; grad: number }[] = [];
  private lastStream: { name: string; grad: number }[] = [];
  private lossEma = NaN;
  constructor(public c: TrainConfig, public tok: Tokenizer, public train: Uint8Array, public val: Uint8Array,
    public prompts: string[] = PROMPTS, public probe: string = PROBE) {
    this.model = new GPT(c, tok.vocab.length);
    this.opt = new Optimizer(c.optimizer);
    this.r = rng(c.seed * 7919 + 3);
    const er = rng(1234), span = Math.max(1, val.length - c.nCtx - 1);
    this.evalIdx = Array.from({ length: 64 }, () => Math.floor(er() * span));
    this.initW = this.weights();
  }
  /** Live-tunable settings (no rebuild). */
  setLive(c: TrainConfig) {
    if (c.optimizer !== this.c.optimizer) this.opt = new Optimizer(c.optimizer);
    this.c = { ...this.c, optimizer: c.optimizer, lr: c.lr, batch: c.batch, clip: c.clip };
  }
  lrNow() { return this.c.lr * Math.min(1, (this.step + 1) / WARMUP); }

  private batch(data: Uint8Array, B: number, T: number, starts?: number[]) {
    const x = new Int32Array(B * T), y = new Int32Array(B * T), span = data.length - T - 1;
    for (let b = 0; b < B; b++) {
      const s = starts ? starts[b] % span : Math.floor(this.r() * span);
      for (let t = 0; t < T; t++) { x[b * T + t] = data[s + t]; y[b * T + t] = data[s + t + 1]; }
    }
    return { x, y };
  }

  trainStep(): number {
    const { batch: B, nCtx: T } = this.c, m = this.model;
    const { x, y } = this.batch(this.train, B, T);
    for (const p of m.params) p.t.grad = null;
    const f = recording(() => { const f = m.forward(x, B, T, y); backward(f.loss!); return f; });
    const loss = f.loss!.data[0];
    // gradient norms (for the "watch gradients flow" bars)
    const g = new Map<string, number>(); let tot = 0;
    for (const p of m.params) { const n = l2(p.t.grad) ** 2; g.set(p.group, (g.get(p.group) ?? 0) + n); tot += n; }
    this.lastGrads = [...g].map(([name, s]) => ({ name, grad: Math.sqrt(s) }));
    this.lastStream = f.stream.map((t, i) => ({ name: i === 0 ? "embed" : `after L${i}`, grad: l2(t.grad) }));
    const norm = Math.sqrt(tot);
    if (this.c.clip && norm > 1) { const s = 1 / norm; for (const p of m.params) if (p.t.grad) for (let i = 0; i < p.t.grad.length; i++) p.t.grad[i] *= s; }
    this.opt.step(m.params, this.lrNow());
    this.step++; this.tokensSeen += B * T; this.lastLoss = loss;
    // diverged = not a number, or (smoothed) far worse than random guessing
    this.lossEma = isNaN(this.lossEma) ? loss : 0.9 * this.lossEma + 0.1 * loss;
    if (!isFinite(loss) || this.lossEma > Math.log(this.model.V) + 2) this.diverged = true;
    return loss;
  }

  evalLoss(nBatches = 4): number {
    const T = this.c.nCtx, B = 16; let s = 0;
    noGrad(() => {
      for (let k = 0; k < nBatches; k++) {
        const { x, y } = this.batch(this.val, B, T, this.evalIdx.slice(k * B, k * B + B));
        s += this.model.forward(x, B, T, y).loss!.data[0];
      }
    });
    return s / nBatches;
  }

  private logitsFor(ids: number[]) {
    const ctx = ids.slice(-this.c.nCtx);
    return this.model.forward(ctx, 1, ctx.length, undefined, true);
  }

  samples(n = 24, temp = 0.8): Sample[] {
    const r = rng(this.c.seed * 31 + this.step);
    return noGrad(() => this.prompts.map((prompt) => {
      const ids = this.tok.encode(prompt).map((t) => t.id), start = ids.length;
      for (let i = 0; i < n; i++) {
        const l = this.logitsFor(ids).logits.data; let mx = -Infinity;
        for (const v of l) if (v > mx) mx = v;
        const p = Array.from(l, (v) => Math.exp((v - mx) / temp)), z = p.reduce((a, b) => a + b, 0);
        let u = r() * z, pick = p.length - 1;
        for (let k = 0; k < p.length; k++) { u -= p[k]; if (u <= 0) { pick = k; break; } }
        ids.push(pick);
      }
      return { prompt, text: this.tok.decode(ids.slice(start)) };
    }));
  }

  /** All weight-matrix entries inside the blocks (falls back to the embedding for a block-less model). */
  private weights() {
    const ws = this.model.params.filter((p) => p.decay && p.name !== "tok" && p.name !== "pos").map((p) => p.t.data);
    if (!ws.length) ws.push(this.model.P.tok.data);
    const out = new Float32Array(ws.reduce((a, w) => a + w.length, 0)); let o = 0;
    for (const w of ws) { out.set(w, o); o += w.length; }
    return out;
  }
  private histogram() {
    const now = this.weights(), init = this.initW, std = (w: Float32Array) => { let s = 0; for (const v of w) s += v * v; return Math.sqrt(s / w.length); };
    const sn = std(now), si = std(init), hi = Math.max(0.06, 3.5 * sn, 3.5 * si), lo = -hi, bins = 41;
    const bin = (w: Float32Array) => { const c = new Array(bins).fill(0); for (const v of w) { const b = Math.floor(((v - lo) / (hi - lo)) * bins); c[Math.max(0, Math.min(bins - 1, b))]++; } return c.map((x) => x / w.length); };
    return { lo, hi, now: bin(now), init: bin(init), std: sn, initStd: si };
  }

  viz(): Viz {
    const m = this.model, c = this.c;
    // weight norms and relative update size per group since the last viz
    const wn = new Map<string, number>(), un = new Map<string, number>();
    for (const p of m.params) {
      const prev = this.prevW.get(p.name); let w2 = 0, d2 = 0;
      for (let i = 0; i < p.t.size; i++) { const v = p.t.data[i]; w2 += v * v; if (prev) { const d = v - prev[i]; d2 += d * d; } }
      wn.set(p.group, (wn.get(p.group) ?? 0) + w2); un.set(p.group, (un.get(p.group) ?? 0) + d2);
      this.prevW.set(p.name, p.t.data.slice());
    }
    const groups = m.groups().map((name) => ({
      name, grad: this.lastGrads.find((g) => g.name === name)?.grad ?? 0,
      weight: Math.sqrt(wn.get(name) ?? 0), update: Math.sqrt(un.get(name) ?? 0) / (Math.sqrt(wn.get(name) ?? 0) + 1e-12),
    }));
    // attention + prediction on the probe prompt
    const ids = this.tok.encode(this.probe).map((t) => t.id).slice(-c.nCtx), T = ids.length, H = c.nHead;
    const f = noGrad(() => m.forward(ids, 1, T));
    const last = f.logits.data.subarray((T - 1) * m.V, T * m.V); let mx = -Infinity;
    for (const v of last) if (v > mx) mx = v;
    const e = Array.from(last, (v) => Math.exp(v - mx)), z = e.reduce((a, b) => a + b, 0);
    const probe = e.map((v, id) => ({ id, p: v / z })).sort((a, b) => b.p - a.p).slice(0, 5).map((x) => ({ token: this.tok.vocab[x.id], p: x.p }));
    const redId = this.tok.index.get(" red") ?? -1;
    const layers = f.probs.map((P) => P && Array.from({ length: H }, (_, h) => Array.from({ length: T }, (_, i) => Array.from(P.subarray((h * T + i) * T, (h * T + i) * T + T)))));
    return {
      step: this.step, groups, stream: this.lastStream,
      attn: { tokens: ids.map((i) => this.tok.vocab[i]), layers }, probe, pRed: redId >= 0 ? e[redId] / z : 0,
      hist: this.histogram(),
    };
  }
}

/** For your own text: prompts and a probe sentence taken from the held-out tail (never trained on). */
export function promptsFrom(tok: Tokenizer, valIds: Uint8Array) {
  const text = tok.decode(Array.from(valIds));
  const sents = text.split(/(?<=[.!?\n])\s+/).map((x) => x.trim()).filter((x) => x.split(" ").length >= 6);
  const pick = (i: number) => sents[Math.floor((i / 4) * sents.length)] ?? text.slice(0, 40);
  const prompts = [0, 1, 2].map((i) => pick(i).split(" ").slice(0, 4).join(" "));
  const probe = (sents.find((x) => x.length >= 50 && x.length <= 110) ?? text.slice(0, 90)).trim().split(" ").slice(0, -1).join(" "); // drop the last word: that's what it should predict
  return { prompts, probe };
}

/** Tokenize text and split 90/10 into train/val. */
export function prepare(tok: Tokenizer, text: string) {
  const ids = Uint8Array.from(tok.encode(text).map((t) => t.id));
  const nVal = Math.max(200, Math.floor(ids.length * 0.1));
  return { train: ids.subarray(0, ids.length - nVal), val: ids.subarray(ids.length - nVal), total: ids.length };
}
