/// <reference lib="webworker" />
// Training runs here so the page stays smooth. Protocol: see client.ts.
import { Tokenizer } from "../model/tokenizer";
import { Trainer, prepare, promptsFrom } from "./trainer";
import type { TrainConfig } from "./config";

export type In =
  | { type: "init"; run: number; cfg: TrainConfig; text: string | null }
  | { type: "run"; on: boolean }
  | { type: "step" }
  | { type: "live"; cfg: TrainConfig };

declare const self: DedicatedWorkerGlobalScope;
let tok: Tokenizer | null = null, defaultText: string | null = null;
let tr: Trainer | null = null, run = 0, running = false, elapsed = 0, sps = 0, lastViz = 0;
let lastText: string | null | undefined, data: ReturnType<typeof prepare> | null = null;
const post = (m: Record<string, unknown>) => self.postMessage({ ...m, run });

async function setup(cfg: TrainConfig, text: string | null, myRun: number) {
  if (!tok) {
    const meta = await fetch("/model/meta.json").then((r) => r.json());
    tok = new Tokenizer(meta.vocab, meta.merges);
  }
  if (text === null && defaultText === null) defaultText = await fetch("/train-data/tiny-world.txt").then((r) => { if (!r.ok) throw new Error("training text not found"); return r.text(); });
  if (myRun !== run) return; // a newer init arrived while we were fetching
  if (lastText !== text || !data) { data = prepare(tok, text ?? defaultText!); lastText = text; }
  if (data.train.length < cfg.nCtx + 2 || data.val.length < cfg.nCtx + 2) throw new Error("That text is too short to train on: paste at least a few thousand characters.");
  const own = text !== null ? promptsFrom(tok, data.val) : null;
  tr = new Trainer(cfg, tok, data.train, data.val, own?.prompts, own?.probe);
  elapsed = 0; sps = 0;
  post({ type: "ready", params: tr.model.nParams, groups: tr.model.groups(), tokens: { train: data.train.length, val: data.val.length }, chars: (text ?? defaultText!).length, probe: tr.probe });
  post({ type: "val", step: 0, loss: tr.evalLoss() });
  post({ type: "samples", step: 0, samples: tr.samples() });
  sendViz();
}

function sendViz() { if (tr) { post({ type: "viz", viz: tr.viz() }); lastViz = performance.now(); } }
function afterStep() {
  const t = tr!;
  if (t.step % 25 === 0) post({ type: "val", step: t.step, loss: t.evalLoss() });
  if (t.step % 50 === 0) post({ type: "samples", step: t.step, samples: t.samples() });
}
function progress(losses: number[]) {
  const t = tr!;
  post({ type: "progress", step: t.step, losses, tokensSeen: t.tokensSeen, lr: t.lrNow(), sps, elapsed, running });
}

function loop(myRun: number) {
  if (!running || !tr || myRun !== run) return;
  const t0 = performance.now(), losses: number[] = [];
  let trainMs = 0;
  while (performance.now() - t0 < 70) {
    const s0 = performance.now(); losses.push(tr.trainStep()); trainMs += performance.now() - s0;
    afterStep();
    if (tr.diverged) break;
  }
  const dt = performance.now() - t0; elapsed += dt / 1000;
  const inst = losses.length / (trainMs / 1000);
  sps = sps ? 0.8 * sps + 0.2 * inst : inst;
  if (tr.diverged) { running = false; progress(losses); post({ type: "diverged", step: tr.step, loss: tr.lastLoss }); sendViz(); return; }
  progress(losses);
  if (performance.now() - lastViz > 300) sendViz();
  setTimeout(() => loop(myRun), 0);
}

self.onmessage = async (e: MessageEvent<In>) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      running = false; run = m.run; tr = null;
      await setup(m.cfg, m.text, m.run);
    } else if (m.type === "run") {
      if (!tr) return;
      if (m.on && !running && !tr.diverged) { running = true; loop(run); }
      if (!m.on && running) { running = false; progress([]); sendViz(); }
    } else if (m.type === "step") {
      if (!tr || running || tr.diverged) return;
      const s0 = performance.now(), l = tr.trainStep(); elapsed += (performance.now() - s0) / 1000;
      afterStep(); progress([l]);
      if (tr.diverged) post({ type: "diverged", step: tr.step, loss: tr.lastLoss });
      sendViz();
    } else if (m.type === "live") {
      tr?.setLive(m.cfg);
    }
  } catch (err) {
    running = false;
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
