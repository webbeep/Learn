// Training config shared by the page, the worker and the Architecture lab links.

export type Optim = "sgd" | "momentum" | "adamw";
export type PresetId = "tiny" | "small" | "medium";

export interface TrainConfig {
  preset: PresetId | "custom";
  nLayer: number; dModel: number; nHead: number; nCtx: number; // dMlp = 4·dModel
  attention: boolean; mlp: boolean; pos: boolean; residual: boolean; layernorm: boolean;
  optimizer: Optim; lr: number; batch: number; seed: number; clip: boolean;
}

export const PRESETS: Record<PresetId, { label: string; nLayer: number; dModel: number; nHead: number; nCtx: number; batch: number; blurb: string }> = {
  tiny: { label: "Tiny", nLayer: 1, dModel: 32, nHead: 2, nCtx: 32, batch: 8, blurb: "1 layer · d 32 · 2 heads" },
  small: { label: "Small", nLayer: 2, dModel: 48, nHead: 3, nCtx: 48, batch: 8, blurb: "2 layers · d 48 · 3 heads" },
  medium: { label: "TinyGPT", nLayer: 3, dModel: 64, nHead: 4, nCtx: 64, batch: 4, blurb: "3 layers · d 64 · 4 heads (the real model's shape)" },
};

/** A sensible starting learning rate per optimizer. */
export const DEFAULT_LR: Record<Optim, number> = { adamw: 1e-2, momentum: 0.1, sgd: 1 };
export const OPTIM_LABEL: Record<Optim, string> = { sgd: "SGD", momentum: "SGD + momentum", adamw: "AdamW" };

export const DEFAULT_CONFIG: TrainConfig = {
  preset: "tiny", ...pick(PRESETS.tiny),
  attention: true, mlp: true, pos: true, residual: true, layernorm: true,
  optimizer: "adamw", lr: DEFAULT_LR.adamw, seed: 1, clip: true,
};
function pick(p: (typeof PRESETS)[PresetId]) { return { nLayer: p.nLayer, dModel: p.dModel, nHead: p.nHead, nCtx: p.nCtx, batch: p.batch }; }
export const withPreset = (c: TrainConfig, id: PresetId): TrainConfig => ({ ...c, preset: id, ...pick(PRESETS[id]) });

/** Changing these rebuilds the model from scratch (step 0). The rest apply live. */
export const STRUCTURAL: (keyof TrainConfig)[] = ["nLayer", "dModel", "nHead", "nCtx", "attention", "mlp", "pos", "residual", "layernorm", "seed"];

export function paramCount(c: TrainConfig, V = 256) {
  const D = c.dModel, M = 4 * D;
  let n = V * D + (c.pos ? c.nCtx * D : 0) + (c.layernorm ? 2 * D : 0);
  const attn = (c.layernorm ? 2 * D : 0) + D * 3 * D + 3 * D + D * D + D;
  const mlp = (c.layernorm ? 2 * D : 0) + D * M + M + M * D + D;
  n += c.nLayer * ((c.attention ? attn : 0) + (c.mlp ? mlp : 0));
  return n;
}

const bool = (v: string) => !(v === "0" || v === "false" || v === "off" || v === "no");
const ALIASES: Record<string, keyof TrainConfig> = {
  preset: "preset", size: "preset",
  layers: "nLayer", n_layer: "nLayer", nlayer: "nLayer", nLayer: "nLayer",
  d: "dModel", d_model: "dModel", dmodel: "dModel", dModel: "dModel",
  heads: "nHead", n_head: "nHead", nHead: "nHead",
  ctx: "nCtx", n_ctx: "nCtx", context: "nCtx", nCtx: "nCtx", block_size: "nCtx",
  attention: "attention", attn: "attention",
  mlp: "mlp",
  pos: "pos", positions: "pos", positional: "pos", pos_emb: "pos",
  residual: "residual", resid: "residual",
  layernorm: "layernorm", ln: "layernorm", norm: "layernorm",
  opt: "optimizer", optimizer: "optimizer",
  lr: "lr", learning_rate: "lr",
  batch: "batch", batch_size: "batch",
  seed: "seed", clip: "clip",
};

/** Apply key/value pairs (from a URL query or an ablation config) on top of a base config. */
export function applyParams(base: TrainConfig, params0: Record<string, unknown>): TrainConfig {
  const c: TrainConfig = { ...base };
  // research configs say no_attention: true; turn those into attention: false
  const params: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params0)) {
    if (k.startsWith("no_") && ALIASES[k.slice(3)]) params[k.slice(3)] = !(v === true || (typeof v === "string" && bool(v)) || v === 1);
    else params[k] = v;
  }
  let custom = false;
  const preset = params.preset ?? params.size;
  if (typeof preset === "string" && preset in PRESETS) Object.assign(c, withPreset(c, preset as PresetId));
  for (const [k0, v0] of Object.entries(params)) {
    const k = ALIASES[k0]; if (!k || k === "preset" || v0 == null) continue;
    const v = String(v0);
    switch (k) {
      case "attention": case "mlp": case "pos": case "residual": case "layernorm": case "clip": c[k] = bool(v); break;
      case "optimizer": {
        const o = v.toLowerCase().replace(/[^a-z]/g, "");
        c.optimizer = o.startsWith("adam") ? "adamw" : o.includes("mom") ? "momentum" : o === "sgd" ? "sgd" : c.optimizer; break;
      }
      case "lr": if (isFinite(+v) && +v > 0) c.lr = +v; break;
      default: if (isFinite(+v) && +v > 0) { (c as unknown as Record<string, number>)[k] = Math.round(+v); if (k !== "batch" && k !== "seed") custom = true; }
    }
  }
  // keep shapes legal and browser-sized
  c.nLayer = clampI(c.nLayer, 1, 4); c.dModel = clampI(c.dModel, 8, 96); c.nCtx = clampI(c.nCtx, 8, 64); c.batch = clampI(c.batch, 1, 32);
  c.lr = Math.min(10, Math.max(1e-5, c.lr));
  if (c.dModel % c.nHead) c.nHead = [4, 3, 2, 1].find((h) => c.dModel % h === 0 && c.dModel / h >= 8) ?? 1;
  if (custom) {
    const match = (Object.keys(PRESETS) as PresetId[]).find((id) => { const p = PRESETS[id]; return p.nLayer === c.nLayer && p.dModel === c.dModel && p.nHead === c.nHead && p.nCtx === c.nCtx; });
    c.preset = match ?? "custom";
  }
  return c;
}
const clampI = (x: number, a: number, b: number) => Math.max(a, Math.min(b, Math.round(x)));

/** Parse "#/train?attention=0&lr=0.01" style hashes. */
export function parseHashQuery(hash: string): Record<string, string> | null {
  const i = hash.indexOf("?"); if (i < 0) return null;
  const out: Record<string, string> = {};
  new URLSearchParams(hash.slice(i + 1)).forEach((v, k) => { out[k] = v; });
  return Object.keys(out).length ? out : null;
}

/** Build a #/train link for an arbitrary config-like object (used by the Architecture lab). */
export function trainHref(params: Record<string, unknown>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (ALIASES[k] && v != null && typeof v !== "object") q.set(k, typeof v === "boolean" ? (v ? "1" : "0") : String(v));
  const s = q.toString();
  return s ? `#/train?${s}` : "#/train";
}

export const PROMPTS = ["mia has a red ball. tom", "once upon a time", "q: what does the cat say? a:", "monday tuesday"];
export const PROBE = "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the";
