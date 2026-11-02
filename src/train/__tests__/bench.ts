// Node benchmark + sanity run: npx tsx src/train/__tests__/bench.ts [preset] [steps] [optimizer] [lr] [key=value ...]
import { readFileSync } from "node:fs";
import { Tokenizer } from "../../model/tokenizer";
import { Trainer, prepare } from "../trainer";
import { DEFAULT_CONFIG, DEFAULT_LR, applyParams, paramCount, type Optim } from "../config";

const [preset = "tiny", stepsS = "200", optim = "adamw", lrS, ...rest] = process.argv.slice(2);
const meta = JSON.parse(readFileSync("public/model/meta.json", "utf8"));
const tok = new Tokenizer(meta.vocab, meta.merges);
const text = readFileSync("public/train-data/tiny-world.txt", "utf8");
let t0 = performance.now();
const d = prepare(tok, text);
console.log(`tokenized ${text.length} chars → ${d.total} tokens in ${(performance.now() - t0).toFixed(0)} ms`);
const extra = Object.fromEntries(rest.map((kv) => kv.split("=")));
const c = applyParams(DEFAULT_CONFIG, { preset, opt: optim, lr: lrS ?? DEFAULT_LR[optim as Optim], ...extra });
const tr = new Trainer(c, tok, d.train, d.val);
console.log(`config`, JSON.stringify(c), `params ${tr.model.nParams} (formula ${paramCount(c)})`);
console.log(`step 0 val ${tr.evalLoss().toFixed(3)}`);
t0 = performance.now(); const N = +stepsS; let tl = 0;
for (let s = 1; s <= N; s++) {
  const l = tr.trainStep(); tl = 0.9 * tl + 0.1 * l; if (s === 1) tl = l;
  if (s % 50 === 0 || s === N) console.log(`step ${s} train(ema) ${tl.toFixed(3)} val ${tr.evalLoss().toFixed(3)}  ${(s / ((performance.now() - t0) / 1000)).toFixed(1)} steps/s`);
  if (tr.diverged) { console.log("DIVERGED at", s, l); break; }
}
let t1 = performance.now(); const smp = tr.samples(); console.log(`samples ${(performance.now() - t1).toFixed(0)} ms`, smp);
t1 = performance.now(); const v = tr.viz(); console.log(`viz ${(performance.now() - t1).toFixed(0)} ms`, JSON.stringify({ g: v.groups, s: v.stream, p: v.probe, pRed: v.pRed, std: v.hist.std }));
