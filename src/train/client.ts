// Page-side handle on the training worker. Module-level so a run survives navigating away and back.
import { createStore } from "../lib/store";
import { DEFAULT_CONFIG, STRUCTURAL, type TrainConfig } from "./config";
import type { Sample, Viz } from "./trainer";
import type { In } from "./worker";

export type Status = "loading" | "ready" | "running" | "paused" | "diverged" | "error";
export interface TrainState {
  cfg: TrainConfig; status: Status; error: string;
  dataKind: "tiny" | "custom"; customText: string; chars: number;
  params: number; groups: string[]; tokens: { train: number; val: number };
  step: number; tokensSeen: number; elapsed: number; sps: number; lr: number;
  train: number[]; val: { step: number; loss: number }[];
  samples: { step: number; list: Sample[] } | null;
  history: { step: number; samples: Sample[] }[]; // a few snapshots to compare
  viz: Viz | null; pRed: { step: number; p: number }[];
  appliedQuery: string; probe: string;
}

export const trainer = createStore<TrainState>({
  cfg: DEFAULT_CONFIG, status: "loading", error: "", dataKind: "tiny", customText: "", chars: 0,
  params: 0, groups: [], tokens: { train: 0, val: 0 },
  step: 0, tokensSeen: 0, elapsed: 0, sps: 0, lr: DEFAULT_CONFIG.lr,
  train: [], val: [], samples: null, history: [], viz: null, pRed: [], appliedQuery: "", probe: "",
});

let worker: Worker | null = null, run = 0, wantRun = false;
const send = (m: In) => worker?.postMessage(m);

function ensureWorker() {
  if (worker) return;
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent) => {
    const m = e.data; if (m.run !== run) return;
    switch (m.type) {
      case "ready":
        trainer.set({ status: "ready", params: m.params, groups: m.groups, tokens: m.tokens, chars: m.chars, probe: m.probe });
        if (wantRun) { wantRun = false; start(); }
        break;
      case "progress":
        trainer.set((s) => ({
          step: m.step, tokensSeen: m.tokensSeen, elapsed: m.elapsed, sps: m.sps, lr: m.lr,
          train: m.losses.length ? s.train.concat(m.losses) : s.train,
        }));
        break;
      case "val": trainer.set((s) => ({ val: [...s.val.filter((v) => v.step !== m.step), { step: m.step, loss: m.loss }] })); break;
      case "samples": trainer.set((s) => {
        const keep = [0, 50, 100, 200, 400, 800, 1600, 3200].includes(m.step);
        return { samples: { step: m.step, list: m.samples }, history: keep ? [...s.history.filter((h) => h.step !== m.step), { step: m.step, samples: m.samples }] : s.history };
      }); break;
      case "viz": trainer.set((s) => ({ viz: m.viz, pRed: s.pRed.length && s.pRed[s.pRed.length - 1].step === m.viz.step ? s.pRed : [...s.pRed, { step: m.viz.step, p: m.viz.pRed }] })); break;
      case "diverged": trainer.set({ status: "diverged" }); break;
      case "error": trainer.set({ status: "error", error: m.message }); break;
    }
  };
  worker.onerror = (e) => trainer.set({ status: "error", error: e.message || "The training worker crashed." });
}

/** (Re)build the model from scratch with the current (or given) config. */
export function reset(cfg = trainer.get().cfg, keepRunning = false) {
  ensureWorker();
  run++; wantRun = keepRunning;
  const s = trainer.get();
  trainer.set({
    cfg, status: "loading", error: "", step: 0, tokensSeen: 0, elapsed: 0, sps: 0, lr: cfg.lr,
    train: [], val: [], samples: null, history: [], viz: null, pRed: [],
  });
  send({ type: "init", run, cfg, text: s.dataKind === "custom" ? s.customText : null });
}
export function start() {
  const s = trainer.get();
  if (s.status === "loading") { wantRun = true; return; }
  if (s.status === "diverged" || s.status === "error") return;
  send({ type: "run", on: true }); trainer.set({ status: "running" });
}
export function pause() {
  wantRun = false;
  const s = trainer.get(); if (s.status !== "running") return;
  send({ type: "run", on: false }); trainer.set({ status: "paused" });
}
export function stepOnce() {
  const s = trainer.get(); if (s.status !== "ready" && s.status !== "paused") return;
  send({ type: "step" }); trainer.set({ status: "paused" });
}
/** Change settings: structural ones restart the run, the rest apply live. Returns true if it restarted. */
export function setCfg(patch: Partial<TrainConfig>): boolean {
  const s = trainer.get(), cfg = { ...s.cfg, ...patch };
  const structural = STRUCTURAL.some((k) => patch[k] !== undefined && patch[k] !== s.cfg[k]);
  if (structural || s.status === "error") { reset(cfg, s.status === "running"); return true; }
  trainer.set({ cfg }); send({ type: "live", cfg });
  return false;
}
export function setData(kind: "tiny" | "custom", text?: string) {
  const s = trainer.get();
  trainer.set({ dataKind: kind, customText: text ?? s.customText });
  reset(s.cfg, s.status === "running");
}
export function ensureStarted() { if (!worker) reset(); }
