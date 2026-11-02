"""Helpers for the 'concepts' videos (backprop, tiny-to-chatgpt, research-loop, frontier).

* TinyGPT: the exact module from train/train.py (copied, because train.py trains at import time),
  loadable with the published weights from public/model/ (via model.W).
* backprop_data(): REAL loss / gradients of TinyGPT on the probe prompt (verified by finite differences).
* toy_graph(): the toy computational graph used to explain the chain rule (verified by finite differences).
* kv_cache_check(): REAL check that cached keys/values give the same logits as recomputing everything.
* corpus_tokens(): the training corpus re-encoded with the published BPE (memoised per pre-token).
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))
from model import W, META, VOCAB, MERGES, tok  # noqa: E402
from bpe import pretokenize  # noqa: E402

PROBE = "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the"
DATA = os.path.join(HERE, "data")


# ---------------------------------------------------------------- model (same as train/train.py)
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
    def __init__(s, c, use_pos=True):
        super().__init__()
        s.c = c; s.use_pos = use_pos
        s.tok = nn.Embedding(c["vocab_size"], c["d_model"]); s.pos = nn.Embedding(c["n_ctx"], c["d_model"])
        s.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layer"])); s.ln_f = nn.LayerNorm(c["d_model"])
        for p in s.parameters():
            if p.dim() == 2: nn.init.normal_(p, std=0.02)

    def forward(s, idx):
        x = s.tok(idx)
        if s.use_pos:
            x = x + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x)
        return s.ln_f(x) @ s.tok.weight.T  # weight tying: unembedding = embedding^T


def load_tiny(dtype=torch.float32):
    m = TinyGPT(META["config"])
    m.load_state_dict({k: v.clone() for k, v in W.items()})
    return m.to(dtype).eval()


# ---------------------------------------------------------------- toy chain-rule graph
TOY = dict(x=2.0, w=3.0, b=1.0, y=5.0)


def toy_forward(x, w, b, y):
    a = w * x
    z = a + b
    L = (z - y) ** 2
    return a, z, L


def toy_graph():
    x, w, b, y = TOY["x"], TOY["w"], TOY["b"], TOY["y"]
    a, z, L = toy_forward(x, w, b, y)
    dL_dz = 2 * (z - y)
    dL_da = dL_dz * 1.0
    dL_db = dL_dz * 1.0
    dL_dw = dL_da * x
    dL_dx = dL_da * w
    eps = 1e-4
    fd = {}
    for k in ("x", "w", "b"):
        hi = dict(TOY); lo = dict(TOY); hi[k] += eps; lo[k] -= eps
        fd[k] = (toy_forward(**hi)[2] - toy_forward(**lo)[2]) / (2 * eps)
    an = {"x": dL_dx, "w": dL_dw, "b": dL_db}
    for k in an:
        assert abs(fd[k] - an[k]) < 1e-6, (k, fd[k], an[k])
    return dict(a=a, z=z, L=L, dz=dL_dz, da=dL_da, db=dL_db, dw=dL_dw, dx=dL_dx, fd=fd, eps=eps)


# ---------------------------------------------------------------- real TinyGPT gradients
def backprop_data(lr=3e-3):
    ids = tok(PROBE)
    tgt = VOCAB.index(" red")
    m = load_tiny(torch.float64)
    x = torch.tensor([ids])
    logits = m(x)[0, -1]
    logits.retain_grad()
    loss = F.cross_entropy(logits[None], torch.tensor([tgt]))
    loss.backward()
    p = logits.detach().softmax(-1)
    g_logit = logits.grad.detach().clone()
    onehot = torch.zeros_like(p); onehot[tgt] = 1
    assert torch.allclose(g_logit, p - onehot, atol=1e-12)

    # gradient norms, grouped by where the parameters live
    groups = {}
    for n, prm in m.named_parameters():
        if n.startswith("blocks."):
            l = int(n.split(".")[1]); part = "attention" if ("attn" in n or "ln1" in n) else "MLP"
            key = f"layer {l + 1} {part}"
        elif n == "tok.weight":
            key = "token embedding (tied)"
        elif n == "pos.weight":
            key = "position embedding"
        else:
            key = "final LayerNorm"
        groups.setdefault(key, 0.0)
        groups[key] += float((prm.grad ** 2).sum())
    norms = {k: math.sqrt(v) for k, v in groups.items()}
    total = math.sqrt(sum(groups.values()))
    n_params = sum(p_.numel() for p_ in m.parameters())
    n_nonzero = sum(int((p_.grad != 0).sum()) for p_ in m.parameters())

    # one real weight: the biggest gradient inside layer 3's MLP output matrix
    name = "blocks.2.mlp_out.weight"
    G = dict(m.named_parameters())[name].grad
    i, j = np.unravel_index(int(G.abs().argmax()), tuple(G.shape))
    i, j = int(i), int(j)
    w0 = float(W[name][i, j]); g = float(G[i, j])
    # finite-difference check of that one gradient (float64)
    eps = 1e-5
    def loss_with(delta):
        mm = load_tiny(torch.float64)
        with torch.no_grad():
            dict(mm.named_parameters())[name][i, j] += delta
            lg = mm(x)[0, -1]
        return float(F.cross_entropy(lg[None], torch.tensor([tgt])))
    fd = (loss_with(eps) - loss_with(-eps)) / (2 * eps)
    assert abs(fd - g) < 1e-6 * max(1, abs(g)), (fd, g)

    # one plain gradient-descent step on ALL weights at lr: what happens to p(red)?
    m2 = load_tiny(torch.float64)
    with torch.no_grad():
        for (n2, p2), (_, p1) in zip(m2.named_parameters(), m.named_parameters()):
            p2 -= lr * p1.grad
        p_after = float(m2(x)[0, -1].softmax(-1)[tgt])
        loss_after = -math.log(p_after)
    # the same step after clipping the whole gradient to length 1 (what train/train.py does before every update)
    m3 = load_tiny(torch.float64)
    sc = min(1.0, 1.0 / total)
    with torch.no_grad():
        for p3, p1 in zip(m3.parameters(), m.parameters()):
            p3 -= lr * sc * p1.grad
        p_after_clip = float(m3(x)[0, -1].softmax(-1)[tgt])

    top = torch.argsort(p, descending=True)[:6].tolist()
    return dict(ids=ids, toks=[VOCAB[t] for t in ids], loss=float(loss.detach()), p_red=float(p[tgt]),
                top=[(VOCAB[t], float(p[t]), float(g_logit[t])) for t in top],
                g_rest_max=float(max(g_logit[t] for t in range(len(VOCAB)) if t not in top)),
                g_sum=float(g_logit.sum()),
                norms=norms, total=total, n_params=n_params, n_nonzero=n_nonzero,
                weight=dict(name=name, i=i, j=j, w=w0, g=g, fd=fd, lr=lr, step=-lr * g, w_new=w0 - lr * g),
                p_after=p_after, loss_after=loss_after, p_after_clip=p_after_clip)


# ---------------------------------------------------------------- KV cache check (real weights)
@torch.no_grad()
def kv_cache_check(prompt=PROBE, n_new=8):
    """Generate greedily with and without a KV cache; return max |logit diff| and op counts."""
    from model import forward, _ln, _lin, C, D, H, HD
    ids = tok(prompt)
    # without cache: full recompute every step
    full_ids = list(ids); full_logits = []
    for _ in range(n_new):
        lg = forward(full_ids)[-1]
        full_logits.append(lg); full_ids.append(int(lg.argmax()))
    # with cache: process one position at a time, keeping K and V per layer
    K = [[] for _ in range(C["n_layer"])]; V = [[] for _ in range(C["n_layer"])]
    def step(t_id, pos):
        x = W["tok.weight"][t_id] + W["pos.weight"][pos]
        for l in range(C["n_layer"]):
            p = f"blocks.{l}."
            q, k, v = _lin(_ln(x[None], p + "ln1"), p + "attn_qkv")[0].split(D, -1)
            K[l].append(k.view(H, HD)); V[l].append(v.view(H, HD))
            Ks = torch.stack(K[l], 1); Vs = torch.stack(V[l], 1)          # H, t, HD
            s = (Ks @ q.view(H, HD, 1))[..., 0] / HD ** 0.5                # H, t
            o = (s.softmax(-1)[..., None] * Vs).sum(1).reshape(D)
            x = x + _lin(o[None], p + "attn_out")[0]
            x = x + _lin(F.gelu(_lin(_ln(x[None], p + "ln2"), p + "mlp_in"), approximate="tanh"), p + "mlp_out")[0]
        return (_ln(x[None], "ln_f") @ W["tok.weight"].T)[0]
    cur = list(ids); diffs = []
    for pos, t in enumerate(ids[:-1]):
        step(t, pos)
    lg = step(ids[-1], len(ids) - 1)
    for k in range(n_new):
        diffs.append(float((lg - full_logits[k]).abs().max()))
        nxt = int(lg.argmax()); cur.append(nxt)
        if k < n_new - 1:
            lg = step(nxt, len(cur) - 1)
    assert cur == full_ids
    T0 = len(ids)
    no_cache = sum(T0 + k for k in range(n_new))   # positions pushed through the network
    with_cache = T0 + (n_new - 1)
    return dict(max_diff=max(diffs), n_prompt=T0, n_new=n_new, text="".join(VOCAB[t] for t in cur[T0:]),
                no_cache=no_cache, with_cache=with_cache)


# ---------------------------------------------------------------- corpus
def corpus_tokens():
    """The full training corpus, re-encoded with the published BPE (cached as .npy)."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(HERE, ".cache", "corpus_tokens.npy")
    if os.path.exists(path):
        return np.load(path)
    from corpus import make_corpus
    from bpe import encode
    text = make_corpus()
    memo = {}
    ids = []
    for w in pretokenize(text):
        if w not in memo:
            memo[w] = encode(w, VOCAB, MERGES)
        ids.extend(memo[w])
    a = np.array(ids, dtype=np.int16)
    np.save(path, a)
    return a


if __name__ == "__main__":
    print(json.dumps(toy_graph(), indent=1))
    d = backprop_data()
    d.pop("ids")
    print(json.dumps(d, indent=1, default=str))
    print(kv_cache_check())
