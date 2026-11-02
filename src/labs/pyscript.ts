// Generates a complete, runnable single-file nanoGPT-style train.py for an experiment card.

export type ChangeId = "baseline" | "no-pos" | "rope" | "deeper" | "shallower" | "no-mlp" | "moe2" | "relu" | "silu" | "custom";

export const CHANGES: { id: ChangeId; label: string; what: string; layers?: number }[] = [
  { id: "baseline", label: "Baseline (no change)", what: "The reference run: TinyGPT's recipe, untouched." },
  { id: "no-pos", label: "No positional embeddings", what: "Remove the learned position vectors. Only the causal mask tells tokens about order." },
  { id: "rope", label: "RoPE instead of learned positions", what: "Rotate queries and keys by an angle that depends on position (rotary embeddings)." },
  { id: "deeper", label: "More layers (6)", what: "Double the depth from 3 to 6 blocks.", layers: 6 },
  { id: "shallower", label: "Fewer layers (1)", what: "A single block.", layers: 1 },
  { id: "no-mlp", label: "No MLP", what: "Attention only: delete the feed-forward sublayer in every block." },
  { id: "moe2", label: "MoE-lite: 2 experts", what: "Two MLPs per block; a learned router sends each token to one of them (top-1, Switch-style)." },
  { id: "relu", label: "ReLU activation", what: "Swap GELU for ReLU inside the MLP." },
  { id: "silu", label: "SiLU (Swish) activation", what: "Swap GELU for SiLU inside the MLP." },
  { id: "custom", label: "Something else (describe it)", what: "Your own idea. The script is the baseline with a marked spot to implement it." },
];

export interface ExperimentCard {
  id: string; created: number; updated: number;
  title: string; hypothesis: string; change: ChangeId; custom: string;
  metric: string; seeds: number; steps: number; notes: string;
  results: { seed: number; baseline: string; variant: string }[];
}

const py = (s: string) => s.replace(/\\/g, "\\\\").replace(/"""/g, '\\"\\"\\"');
const oneLine = (s: string) => s.replace(/\s+/g, " ").trim();

export function makeScript(card: ExperimentCard): string {
  const ch = card.change;
  const meta = CHANGES.find((c) => c.id === ch)!;
  const nLayer = meta.layers ?? 3;
  const pos = ch === "no-pos" ? "none" : ch === "rope" ? "rope" : "learned";
  const act = ch === "relu" ? "relu" : ch === "silu" ? "silu" : "gelu";
  const mlp = ch === "no-mlp" ? "none" : ch === "moe2" ? "moe" : "dense";
  const seeds = Array.from({ length: Math.max(1, card.seeds) }, (_, i) => i).join(" ");
  const actExpr = act === "relu" ? "F.relu(h)" : act === "silu" ? "F.silu(h)" : 'F.gelu(h, approximate="tanh")';
  const mark = (on: boolean) => (on ? "  # <<< EXPERIMENT" : "");

  const header = `"""${py(card.title || "My experiment")}

Experiment card (exported from Top-Down AI, "Invent your own")
  Hypothesis: ${py(oneLine(card.hypothesis) || "(write one!)")}
  Change:     ${py(ch === "custom" ? oneLine(card.custom) || "custom change" : meta.label)}
  Metric:     ${py(card.metric)}
  Seeds:      ${seeds}

A complete, single-file GPT in the style of nanoGPT / TinyGPT: pre-LayerNorm blocks, causal self-attention,
weight-tied unembedding, AdamW with warmup + cosine decay, gradient clipping. Character-level tokens, so it
runs on any text file without extra dependencies (TinyGPT itself uses a 256-token BPE).

Lines whose comment says EXPERIMENT are the ones your change touches. Run the baseline too and compare!

Make the tiny-world training text (from the Learn repo root):
    python3 -c "import sys; sys.path.insert(0, 'train'); from corpus import make_corpus; open('tiny_world.txt', 'w').write(make_corpus())"
Then:
    python3 train.py --data tiny_world.txt                 # ${card.steps} steps × ${card.seeds} seed(s)
    python3 train.py --data tiny_world.txt --steps 200     # quick smoke test
Requires: Python 3.9+, PyTorch 2.x.
"""
import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENT = ${JSON.stringify(ch)}
CONFIG = dict(
    n_layer=${nLayer},${mark(nLayer !== 3)}
    n_head=4,
    d_model=64,
    d_mlp=256,
    n_ctx=64,
    positions=${JSON.stringify(pos)},${mark(pos !== "learned")}  # "learned" | "rope" | "none"
    mlp=${JSON.stringify(mlp)},${mark(mlp !== "dense")}  # "dense" | "moe" | "none"
    activation=${JSON.stringify(act)},${mark(act !== "gelu")}  # "gelu" | "relu" | "silu"
    n_experts=2,
    dropout=0.0,
)
`;

  const ropeCode = pos === "rope" ? `

def apply_rope(q, k):  # <<< EXPERIMENT: rotary position embeddings (Su et al. 2021)
    """Rotate each (even, odd) pair of q/k features by an angle proportional to the position."""
    T, hd = q.shape[-2], q.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=q.device).float() / hd))
    ang = torch.arange(T, device=q.device).float()[:, None] * inv_freq[None, :]  # (T, hd/2)
    cos, sin = ang.cos(), ang.sin()

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    return rot(q), rot(k)
` : "";

  const attn = `

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head = c["n_head"]
        self.qkv = nn.Linear(c["d_model"], 3 * c["d_model"])
        self.out = nn.Linear(c["d_model"], c["d_model"])

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))${pos === "rope" ? `
        q, k = apply_rope(q, k)  # <<< EXPERIMENT` : ""}
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.fc_out = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        h = self.fc_in(x)
        h = ${actExpr}${mark(act !== "gelu")}
        return self.fc_out(h)
`;

  const moe = mlp === "moe" ? `

class MoELite(nn.Module):  # <<< EXPERIMENT: mixture of experts (Switch Transformer style, top-1)
    """n_experts MLPs; a router sends every token to the single expert it scores highest.
    Parameters grow ~n_experts×, but each token still runs through only one MLP."""

    def __init__(self, d_model, d_mlp, n_experts):
        super().__init__()
        self.n = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(MLP(d_model, d_mlp) for _ in range(n_experts))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        probs = F.softmax(self.router(x), dim=-1)  # (B, T, E)
        top_p, top_i = probs.max(dim=-1)  # (B, T)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = top_i == e
            if mask.any():
                out[mask] = expert(x[mask]) * top_p[mask].unsqueeze(-1)
        # load-balancing loss: (fraction of tokens sent to e) · (mean router prob of e), summed, × n
        frac = torch.stack([(top_i == e).float().mean() for e in range(self.n)])
        self.aux_loss = self.n * (frac * probs.mean(dim=(0, 1))).sum()
        return out
` : "";

  const mlpInit = mlp === "none" ? `        self.ffn = None  # <<< EXPERIMENT: no MLP, attention only`
    : mlp === "moe" ? `        self.ln2 = nn.LayerNorm(c["d_model"])
        self.ffn = MoELite(c["d_model"], c["d_mlp"], c["n_experts"])  # <<< EXPERIMENT`
    : `        self.ln2 = nn.LayerNorm(c["d_model"])
        self.ffn = MLP(c["d_model"], c["d_mlp"])`;

  const block = `

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c["d_model"])
        self.attn = CausalSelfAttention(c)
${mlpInit}

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.ffn is not None:
            x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c, vocab_size):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(vocab_size, c["d_model"])
${pos === "learned" ? `        self.pos = nn.Embedding(c["n_ctx"], c["d_model"])` : `        self.pos = None  # <<< EXPERIMENT: no learned position embeddings`}
        self.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layer"]))
        self.ln_f = nn.LayerNorm(c["d_model"])
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.ln_f(x) @ self.tok.weight.T  # weight tying: unembedding = embedding^T
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def aux_loss(self):
        return sum((getattr(b.ffn, "aux_loss", 0.0) for b in self.blocks), torch.tensor(0.0))

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.c["n_ctx"]:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
`;

  const train = `

# ---------------------------------------------------------------------------- data
def load_text(path):
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n_val = len(data) // 20  # last 5% held out, like TinyGPT
    return data[:-n_val], data[-n_val:], chars, stoi


def get_batch(data, batch, n_ctx, gen, device):
    ix = torch.randint(len(data) - n_ctx - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + n_ctx] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + n_ctx] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, args, device, n_batches=20):
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # same eval batches for every run
    losses = [model(*get_batch(data, args.batch, CONFIG["n_ctx"], gen, device))[1].item() for _ in range(n_batches)]
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------- training
def train_one(seed, train_data, val_data, vocab, args, device):
    torch.manual_seed(seed)
    model = GPT(CONFIG, len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    gen = torch.Generator().manual_seed(seed)
    curve, t0 = [], time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate(model, val_data, args, device)
            curve.append({"step": step, "val": round(vl, 4)})
            print(f"  seed {seed}  step {step:5d}  val {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if step == args.steps:
            break
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_data, args.batch, CONFIG["n_ctx"], gen, device)
        _, loss = model(x, y)${mlp === "moe" ? `
        loss = loss + 0.01 * model.aux_loss()  # <<< EXPERIMENT: keep both experts busy` : ""}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    secs = time.time() - t0
    return model, {"seed": seed, "params": n_params, "final_val": curve[-1]["val"], "seconds": round(secs, 1),
                   "ms_per_step": round(1000 * secs / max(1, args.steps), 2), "curve": curve}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to a UTF-8 text file to train on")
    ap.add_argument("--steps", type=int, default=${card.steps})
    ap.add_argument("--seeds", type=int, nargs="+", default=[${seeds.split(" ").join(", ")}])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results_${ch}.json")
    args = ap.parse_args()
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_data, val_data, vocab, stoi = load_text(args.data)
    print(f"experiment: {EXPERIMENT}  device: {device}  vocab: {len(vocab)} chars  tokens: {len(train_data):,} train / {len(val_data):,} val")
    results, model = [], None
    for seed in args.seeds:
        model, r = train_one(seed, train_data, val_data, vocab, args, device)
        results.append(r)
        print(f"seed {seed}: final val loss {r['final_val']:.4f}  params {r['params']:,}  {r['ms_per_step']} ms/step")

    vals = [r["final_val"] for r in results]
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    print(f"\\n== {EXPERIMENT}: val loss {mean:.4f} ± {std:.4f} over {len(vals)} seed(s)  (metric: ${py(card.metric).replace(/"/g, "'").replace(/[{}]/g, "")})")

    prompt = "once upon a time"
    idx = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], device=device)
    torch.manual_seed(1)
    sample = "".join(vocab[i] for i in model.generate(idx, 120)[0].tolist())
    print(f"\\nsample: {sample!r}")
    json.dump({"experiment": EXPERIMENT, "config": CONFIG, "mean_val": mean, "std_val": std, "runs": results, "sample": sample},
              open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
`;

  const customNote = ch === "custom" ? `

# <<< EXPERIMENT: implement your idea here, then mark the lines you change.
# Your description: ${oneLine(card.custom || "(none yet)").replace(/\n/g, " ")}
# Tip: change exactly one thing, keep everything else identical to the baseline, and compare across seeds.
` : "";

  return header + customNote + ropeCode + attn + moe + block + train;
}

export function scriptName(card: ExperimentCard) {
  const slug = (card.title || card.change).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 40) || "experiment";
  return `train_${slug}.py`;
}

/** Changes the in-browser Train lab (#/train) can reproduce, as its query parameters. */
const BROWSER: Partial<Record<ChangeId, Record<string, string>>> = {
  baseline: {}, "no-pos": { pos: "0" }, "no-mlp": { mlp: "0" }, deeper: { layers: "4" }, shallower: { layers: "1" },
};
export const browserSupports = (c: ChangeId) => c in BROWSER;

/** Deep link into the in-browser Train lab with this card's change (the baseline if the lab can't do that change). */
export function browserLink(card: ExperimentCard) {
  const q = new URLSearchParams({ preset: "small", seed: "1", ...(BROWSER[card.change] ?? {}) });
  return `#/train?${q.toString()}`;
}
