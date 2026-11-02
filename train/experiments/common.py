"""Shared pieces for the research experiments.

* Same tokenizer as the shipped model: vocab + merges are loaded from public/model/meta.json and the
  corpus is encoded with train/bpe.encode, so token ids match the browser model exactly.
* Same train/val split as train/train.py (last 5% of tokens = validation).
* A configurable TinyGPT (copied from train/train.py) with switches to remove parts.
* One training loop used by every experiment, evaluated on fixed held-out batches.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "train"))
from corpus import make_corpus  # noqa: E402
from bpe import encode  # noqa: E402

META = json.load(open(os.path.join(ROOT, "public", "model", "meta.json")))
VOCAB, MERGES = META["vocab"], [tuple(m) for m in META["merges"]]
OUT = os.path.join(ROOT, "public", "research")
CACHE = os.path.join(HERE, ".cache")

PROBE = "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the"
PROMPTS = [PROBE, "once upon a time", "q: what does the cat say? a:"]

BASE_CFG = dict(vocab_size=len(VOCAB), n_ctx=64, d_model=64, n_head=4, n_layer=3, d_mlp=256,
                no_attention=False, no_mlp=False, no_positions=False, no_residual=False, no_layernorm=False)


def load_tokens():
    """Encode the corpus once (≈1 min) and cache it as int16."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "tokens.npy")
    if not os.path.exists(path):
        ids = encode(make_corpus(), VOCAB, list(MERGES))
        np.save(path, np.array(ids, dtype=np.int16))
    data = torch.from_numpy(np.load(path).astype(np.int64))
    n_val = len(data) // 20
    return data[:-n_val], data[-n_val:]


def tok_ids(text): return encode(text, VOCAB, list(MERGES))
def decode(ids): return "".join(VOCAB[i] for i in ids)


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        D = c["d_model"]; LN = nn.Identity if c["no_layernorm"] else nn.LayerNorm
        s.c = c; s.h = c["n_head"]
        s.ln1 = LN(D); s.attn_qkv = nn.Linear(D, 3 * D); s.attn_out = nn.Linear(D, D)
        s.ln2 = LN(D); s.mlp_in = nn.Linear(D, c["d_mlp"]); s.mlp_out = nn.Linear(c["d_mlp"], D)

    def attn(s, x):
        B, T, D = x.shape; hd = D // s.h
        q, k, v = s.attn_qkv(s.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, s.h, hd).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.attn_out(a.transpose(1, 2).reshape(B, T, D))

    def mlp(s, x): return s.mlp_out(F.gelu(s.mlp_in(s.ln2(x)), approximate="tanh"))

    def forward(s, x):
        res = not s.c["no_residual"]
        if not s.c["no_attention"]:
            x = x + s.attn(x) if res else s.attn(x)
        if not s.c["no_mlp"]:
            x = x + s.mlp(x) if res else s.mlp(x)
        return x


class TinyGPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c = {**BASE_CFG, **c}
        s.tok = nn.Embedding(c["vocab_size"], c["d_model"]); s.pos = nn.Embedding(c["n_ctx"], c["d_model"])
        s.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layer"]))
        s.ln_f = nn.Identity() if c["no_layernorm"] else nn.LayerNorm(c["d_model"])
        for p in s.parameters():
            if p.dim() == 2: nn.init.normal_(p, std=0.02)
        if c["no_positions"]:
            s.pos.weight.requires_grad_(False); nn.init.zeros_(s.pos.weight)

    def forward(s, idx):
        x = s.tok(idx)
        if not s.c["no_positions"]: x = x + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x)
        return s.ln_f(x) @ s.tok.weight.T

    def n_params(s):
        return sum(p.numel() for n, p in s.named_parameters() if p.requires_grad and not
                   (n.startswith("blocks.") and _unused(s.c, n)))

    def n_non_embedding(s):
        return s.n_params() - sum(p.numel() for n, p in s.named_parameters()
                                  if n.startswith(("tok.", "pos.")) and p.requires_grad)


def _unused(c, n):
    return (c["no_attention"] and (".ln1" in n or ".attn" in n)) or (c["no_mlp"] and (".ln2" in n or ".mlp" in n))


def get_batch(d, B, T, gen):
    ix = torch.randint(len(d) - T - 1, (B,), generator=gen)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + 1 + T] for i in ix])


def fixed_batches(d, n, B, T, seed):
    g = torch.Generator().manual_seed(seed)
    return [get_batch(d, B, T, g) for _ in range(n)]


@torch.no_grad()
def eval_loss(m, batches):
    m.eval(); tot = 0.0
    for x, y in batches:
        tot += F.cross_entropy(m(x).view(-1, m.c["vocab_size"]), y.view(-1)).item()
    m.train(); return tot / len(batches)


@torch.no_grad()
def sample(m, prompt, n=24, temp=0.8, seed=1):
    m.eval(); g = torch.Generator().manual_seed(seed); ids = tok_ids(prompt)
    for _ in range(n):
        logits = m(torch.tensor([ids[-m.c["n_ctx"]:]]))[0, -1]
        if not torch.isfinite(logits).all(): break
        ids.append(int(torch.multinomial(F.softmax(logits / temp, -1), 1, generator=g)))
    m.train(); return decode(ids)


@torch.no_grad()
def top_next(m, prompt, k=5):
    m.eval(); p = F.softmax(m(torch.tensor([tok_ids(prompt)]))[0, -1], -1); m.train()
    v, i = p.topk(k)
    return [{"token": VOCAB[int(j)], "p": round(float(q), 4)} for q, j in zip(v, i)]


def train_run(cfg, steps=1500, batch=32, lr=3e-3, opt="adamw", momentum=0.0, wd=0.05, warmup=100,
              schedule="cosine", clip=1.0, eval_every=50, train_data=None, seed=0, threads=2,
              n_eval=4, log=None):
    """Train one model; returns dict with model, curve (fixed-batch train/val loss), diverged flag."""
    torch.set_num_threads(threads); torch.manual_seed(seed)
    train_d, val_d = load_tokens()
    if train_data is not None: train_d = train_d[:train_data]
    m = TinyGPT(cfg); T = m.c["n_ctx"]
    params = [p for p in m.parameters() if p.requires_grad]
    if opt == "adamw":
        o = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        o = torch.optim.SGD(params, lr=lr, momentum=momentum)
    val_b = fixed_batches(val_d, n_eval, batch, T, 1234)
    tr_b = fixed_batches(train_d, n_eval, batch, T, 4321)
    gen = torch.Generator().manual_seed(seed)
    curve = {"step": [], "tokens": [], "train": [], "val": []}
    diverged = False; t0 = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            tl, vl = eval_loss(m, tr_b), eval_loss(m, val_b)
            if not (math.isfinite(tl) and math.isfinite(vl)) or vl > 20:
                diverged = True
            curve["step"].append(step); curve["tokens"].append(step * batch * T)
            curve["train"].append(round(tl, 4) if math.isfinite(tl) else None)
            curve["val"].append(round(vl, 4) if math.isfinite(vl) else None)
            if log: log(f"{step:5d} train {tl:.3f} val {vl:.3f} {time.time() - t0:.0f}s")
            if diverged or step == steps: break
        f = min(1.0, (step + 1) / warmup) if warmup else 1.0
        if schedule == "cosine": f *= 0.5 * (1 + math.cos(math.pi * step / steps))
        for g_ in o.param_groups: g_["lr"] = lr * f
        x, y = get_batch(train_d, batch, T, gen)
        loss = F.cross_entropy(m(x).view(-1, m.c["vocab_size"]), y.view(-1))
        o.zero_grad(); loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(params, clip)
        o.step()
    return dict(model=m, curve=curve, diverged=diverged, seconds=round(time.time() - t0, 1))


def finite_last(xs):
    for v in reversed(xs):
        if v is not None: return v
    return None


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    json.dump(obj, open(path, "w"), separators=(",", ":"))
    json.load(open(path))
    print("wrote", path, os.path.getsize(path), "bytes")
