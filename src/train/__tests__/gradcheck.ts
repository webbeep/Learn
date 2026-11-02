// Finite-difference gradient check for every op in src/train/tensor.ts and for the full GPT.
// Run: npx tsx src/train/__tests__/gradcheck.ts
import { Tensor, add, attention, backward, crossEntropy, embedding, gelu, layerNorm, linear, noGrad, recording, rows } from "../tensor";
import { GPT, Optimizer, gauss, rng } from "../model";
import { DEFAULT_CONFIG, type TrainConfig } from "../config";

const r = rng(42);
const rand = (shape: number[], s = 1) => { const t = Tensor.zeros(shape, true); for (let i = 0; i < t.size; i++) t.data[i] = s * gauss(r); return t; };

/** Compare analytic d(sum(out·R))/d(input) with central differences, on up to `samples` entries per input. */
function check(name: string, inputs: Tensor[], f: () => Tensor, samples = 40, eps = 1e-2) {
  let R: Float32Array | null = null;
  const objective = () => { const out = f(); R ??= Float32Array.from({ length: out.size }, () => gauss(r)); let s = 0; for (let i = 0; i < out.size; i++) s += out.data[i] * R[i]; return { out, s }; };
  inputs.forEach((t) => (t.grad = null));
  recording(() => { const { out } = objective(); backward(out, R!); });
  let worst = 0;
  for (const [k, t] of inputs.entries()) {
    const idx = t.size <= samples ? [...Array(t.size).keys()] : Array.from({ length: samples }, () => Math.floor(r() * t.size));
    for (const i of idx) {
      const v = t.data[i];
      t.data[i] = v + eps; const sp = noGrad(objective).s;
      t.data[i] = v - eps; const sm = noGrad(objective).s;
      t.data[i] = v;
      const num = (sp - sm) / (2 * eps), ana = t.grad ? t.grad[i] : 0;
      const err = Math.abs(num - ana) / Math.max(1e-2, Math.abs(num) + Math.abs(ana));
      if (err > worst) worst = err;
      if (err > 2e-2) console.log(`   ${name} input#${k}[${i}] analytic ${ana.toFixed(5)} numeric ${num.toFixed(5)}`);
    }
  }
  const ok = worst < 2e-2;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(34)} max rel err ${worst.toExponential(2)}`);
  if (!ok) process.exitCode = 1;
  return worst;
}
const results: Record<string, number> = {};
{ const W = rand([11, 6]); const ids = [3, 0, 3, 10, 7]; results.embedding = check("embedding (gather)", [W], () => embedding(W, ids)); }
{ const x = rand([5, 7]), W = rand([4, 7]), b = rand([4]); results.linear = check("linear x·Wᵀ+b", [x, W, b], () => linear(x, W, b)); }
{ const x = rand([5, 7]), W = rand([9, 7]); results.linearTied = check("linear without bias (unembed)", [x, W], () => linear(x, W)); }
{ const a = rand([4, 6]), b = rand([4, 6]); results.add = check("add", [a, b], () => add(a, b)); }
{ const a = rand([4, 6]), b = rand([6]); results.addBroadcast = check("add (broadcast row)", [a, b], () => add(a, b)); }
{ const x = rand([6, 5], 2); results.gelu = check("gelu (tanh)", [x], () => gelu(x)); }
{ const x = rand([5, 8], 1.5), g = rand([8]), b = rand([8]); results.layerNorm = check("layerNorm", [x, g, b], () => layerNorm(x, g, b)); }
{ const B = 2, Tn = 5, H = 2, D = 8; const qkv = rand([B * Tn, 3 * D]); results.attention = check("causal self-attention (B2 T5 H2)", [qkv], () => attention(qkv, B, Tn, H).out, 120); }
{ const x = rand([6, 10], 2); const tg = [1, 9, 0, 4, 4, 2]; results.crossEntropy = check("softmax cross-entropy", [x], () => crossEntropy(x, tg)); }
{ const x = rand([6, 4]); results.rows = check("rows (slice)", [x], () => rows(x, 5, 1)); }

// Full model, several architecture toggles: check a sample of every parameter tensor.
const tiny = (over: Partial<TrainConfig>): TrainConfig => ({ ...DEFAULT_CONFIG, nLayer: 2, dModel: 8, nHead: 2, nCtx: 6, ...over });
const cfgs: [string, Partial<TrainConfig>][] = [
  ["GPT full", {}], ["GPT no LayerNorm", { layernorm: false }], ["GPT no residual", { residual: false }],
  ["GPT no positions", { pos: false }], ["GPT attention only", { mlp: false }], ["GPT MLP only", { attention: false }],
];
for (const [name, over] of cfgs) {
  const c = tiny(over), m = new GPT(c, 20);
  for (const p of m.params) for (let i = 0; i < p.t.size; i++) p.t.data[i] += 0.3 * gauss(r); // break symmetry of LN/bias init
  const B = 2, Tn = 6, ids = Array.from({ length: B * Tn }, () => Math.floor(r() * 20)), tg = Array.from({ length: B * Tn }, () => Math.floor(r() * 20));
  results[name] = check(name, m.params.map((p) => p.t), () => m.forward(ids, B, Tn, tg).loss!, 12, 5e-3);
}

// Optimizers reduce the loss on a fixed batch.
for (const kind of ["sgd", "momentum", "adamw"] as const) {
  const m = new GPT(tiny({}), 20), opt = new Optimizer(kind), ids = Array.from({ length: 12 }, (_, i) => i % 7), tg = ids.map((x) => (x + 1) % 7);
  const lr = { sgd: 0.5, momentum: 0.1, adamw: 1e-2 }[kind]; let first = 0, last = 0;
  for (let s = 0; s < 60; s++) {
    m.params.forEach((p) => (p.t.grad = null));
    const loss = recording(() => { const f = m.forward(ids, 2, 6, tg); backward(f.loss!); return f.loss!.data[0]; });
    if (s === 0) first = loss; last = loss; opt.step(m.params, lr);
  }
  const ok = last < first * 0.5; if (!ok) process.exitCode = 1;
  console.log(`${ok ? "PASS" : "FAIL"}  optimizer ${kind.padEnd(24)} loss ${first.toFixed(3)} → ${last.toFixed(3)} in 60 steps`);
}
console.log(JSON.stringify(results));
