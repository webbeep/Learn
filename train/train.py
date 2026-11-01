"""Train TinyGPT on the tiny-world corpus and export weights for the browser.

Run:  python3 train.py      (≈ a few minutes on CPU)
Writes ../public/model/{weights.bin, meta.json, training.json}
"""
import json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from corpus import make_corpus
from bpe import train_bpe, encode

torch.manual_seed(0); random.seed(0)
OUT = os.path.join(os.path.dirname(__file__), "..", "public", "model")
os.makedirs(OUT, exist_ok=True)

CFG = dict(vocab_size=256, n_ctx=64, d_model=64, n_head=4, n_layer=3, d_mlp=256)
STEPS, BATCH, LR = 3000, 64, 3e-3

text = make_corpus()
vocab, merges = train_bpe(text[:400_000], CFG["vocab_size"])
CFG["vocab_size"] = len(vocab)
data = torch.tensor(encode(text, vocab, merges), dtype=torch.long)
print(f"chars={len(text):,} tokens={len(data):,} vocab={len(vocab)}")
n_val = len(data) // 20
train_d, val_d = data[:-n_val], data[-n_val:]


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.ln1 = nn.LayerNorm(c["d_model"]); s.attn_qkv = nn.Linear(c["d_model"], 3 * c["d_model"])
        s.attn_out = nn.Linear(c["d_model"], c["d_model"]); s.ln2 = nn.LayerNorm(c["d_model"])
        s.mlp_in = nn.Linear(c["d_model"], c["d_mlp"]); s.mlp_out = nn.Linear(c["d_mlp"], c["d_model"])
        s.h = c["n_head"]

    def forward(s, x):
        B, T, D = x.shape; hd = D // s.h
        q, k, v = s.attn_qkv(s.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, s.h, hd).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.attn_out(a.transpose(1, 2).reshape(B, T, D))
        return x + s.mlp_out(F.gelu(s.mlp_in(s.ln2(x)), approximate="tanh"))


class TinyGPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.tok = nn.Embedding(c["vocab_size"], c["d_model"]); s.pos = nn.Embedding(c["n_ctx"], c["d_model"])
        s.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layer"])); s.ln_f = nn.LayerNorm(c["d_model"])
        for p in s.parameters():
            if p.dim() == 2: nn.init.normal_(p, std=0.02)

    def forward(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x)
        return s.ln_f(x) @ s.tok.weight.T  # weight tying: unembedding = embedding^T


def batch(d):
    ix = torch.randint(len(d) - CFG["n_ctx"] - 1, (BATCH,))
    x = torch.stack([d[i:i + CFG["n_ctx"]] for i in ix]); y = torch.stack([d[i + 1:i + 1 + CFG["n_ctx"]] for i in ix])
    return x, y


def decode(ids): return "".join(vocab[i] for i in ids)


@torch.no_grad()
def sample(m, prompt, n=24):
    ids = encode(prompt, vocab, merges)
    for _ in range(n):
        p = F.softmax(m(torch.tensor([ids[-CFG["n_ctx"]:]]))[0, -1] / 0.8, -1)
        ids.append(int(torch.multinomial(p, 1)))
    return decode(ids)


m = TinyGPT(CFG)
print("params", sum(p.numel() for p in m.parameters()))
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.05)
PROMPTS = ["mia has a red ball. tom", "once upon a time", "q: what does the cat say? a:", "monday tuesday"]
SNAP = {0, 25, 100, 300, 1000, STEPS}
log = {"steps": [], "train": [], "val": [], "snapshots": []}
t0 = time.time()
for step in range(STEPS + 1):
    if step in SNAP:
        m.eval(); g = torch.Generator().manual_seed(1); torch.manual_seed(1)
        log["snapshots"].append({"step": step, "samples": [sample(m, p) for p in PROMPTS]}); m.train()
    lr = LR * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
    for g_ in opt.param_groups: g_["lr"] = lr
    x, y = batch(train_d)
    loss = F.cross_entropy(m(x).view(-1, CFG["vocab_size"]), y.view(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 25 == 0:
        with torch.no_grad():
            m.eval(); xv, yv = batch(val_d)
            vl = F.cross_entropy(m(xv).view(-1, CFG["vocab_size"]), yv.view(-1)).item(); m.train()
        log["steps"].append(step); log["train"].append(round(loss.item(), 4)); log["val"].append(round(vl, 4))
    if step % 500 == 0:
        print(f"step {step} loss {loss.item():.3f} val {vl:.3f} {time.time() - t0:.0f}s")

m.eval()
for p in PROMPTS: print(repr(sample(m, p, 30)))

# ---- export -------------------------------------------------------------
tensors, offset, blob = [], 0, []
for name, p in m.state_dict().items():
    a = p.detach().numpy().astype("<f4")
    tensors.append({"name": name, "shape": list(a.shape), "offset": offset}); offset += a.size; blob.append(a.ravel())
np.concatenate(blob).tofile(os.path.join(OUT, "weights.bin"))

E = m.tok.weight.detach().numpy(); Ec = E - E.mean(0)
_, _, Vt = np.linalg.svd(Ec, full_matrices=False)
pca = (Ec @ Vt[:2].T)
pca = (pca / np.abs(pca).max()).round(4).tolist()
json.dump({"config": CFG, "vocab": vocab, "merges": merges, "tensors": tensors, "pca": pca,
           "corpus_sample": text[:1500]}, open(os.path.join(OUT, "meta.json"), "w"))
log.update(prompts=PROMPTS, params=sum(p.numel() for p in m.parameters()), tokens=len(data), batch=BATCH, lr=LR)
json.dump(log, open(os.path.join(OUT, "training.json"), "w"))
print("exported", offset, "floats")
