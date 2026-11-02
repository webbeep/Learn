"""Helpers for the inference/loss explainers (embeddings, mlp, residual-stream, loss).

Everything here reads the REAL TinyGPT weights through video/model.py; nothing is hard-coded except
word lists (taken from train/corpus.py) and the probe prompt.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import W, C, D, H, HD, VOCAB, META, TRAIN, tok, forward, lens, _ln, _lin  # noqa: E402,F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import corpus as _corpus  # noqa: E402

PROBE = "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the"
E = W["tok.weight"]            # (256, 64) token embeddings == unembedding (weight tying)
P = W["pos.weight"]            # (64, 64) learned position embeddings
GAMMA = W["ln_f.weight"]


def tid(t):
    return VOCAB.index(t)


# word groups straight from the corpus generator (only those that are single tokens with a leading space)
def _grp(words):
    return [" " + w for w in words if " " + w in VOCAB]


GROUPS = {
    "names": _grp(_corpus.NAMES),
    "colors": _grp(_corpus.COLORS),
    "animals": _grp(_corpus.ANIMALS),
    "objects": _grp(_corpus.OBJECTS),
    "places": _grp(_corpus.PLACES),
}


# --------------------------------------------------------------------------- embeddings
def cos_matrix(M):
    M = M.numpy() if hasattr(M, "numpy") else M
    n = M / np.linalg.norm(M, axis=1, keepdims=True)
    return n @ n.T


def neighbours(t, k=5):
    S = cos_matrix(E)
    i = tid(t)
    o = [j for j in np.argsort(-S[i]) if j != i][:k]
    return [(VOCAB[j], float(S[i, j])) for j in o]


def pca2(M):
    M = M.numpy() if hasattr(M, "numpy") else M
    Mc = M - M.mean(0)
    _, s, Vt = np.linalg.svd(Mc, full_matrices=False)
    Y = Mc @ Vt[:2].T
    return Y, (s ** 2 / (s ** 2).sum())


def group_nn_accuracy():
    """Leave-one-out: for each group word, is its nearest (cosine) group word in the same group?"""
    words = [(w, g) for g, l in GROUPS.items() for w in l]
    S = cos_matrix(E)
    ok = 0
    for w, g in words:
        i = tid(w)
        best = max(((S[i, tid(v)], h) for v, h in words if v != w))
        ok += best[1] == g
    return ok, len(words)


# --------------------------------------------------------------------------- batched forward with full trace
@torch.no_grad()
def run(X, skip=()):
    """X: (B, T) long tensor or list of ids. Returns dict with resid stages, mlp acts (post-GELU), logits."""
    if not torch.is_tensor(X):
        X = torch.tensor(X)
    if X.dim() == 1:
        X = X[None]
    B, T = X.shape
    x = E[X] + P[:T]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    stages = [("embed", x.clone())]
    acts, writes = [], []
    for l in range(C["n_layer"]):
        p = f"blocks.{l}."
        q, k, v = _lin(_ln(x, p + "ln1"), p + "attn_qkv").split(D, -1)
        q, k, v = (t.view(B, T, H, HD).transpose(1, 2) for t in (q, k, v))
        s = (q @ k.transpose(-1, -2) / HD ** 0.5).masked_fill(mask, float("-inf"))
        o = _lin((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D), p + "attn_out")
        if ("attn", l) in skip:
            o = torch.zeros_like(o)
        x = x + o
        writes.append(("attn", l, o))
        stages.append((f"attn{l + 1}", x.clone()))
        h = F.gelu(_lin(_ln(x, p + "ln2"), p + "mlp_in"), approximate="tanh")
        acts.append(h)
        m = _lin(h, p + "mlp_out")
        if ("mlp", l) in skip:
            m = torch.zeros_like(m)
        x = x + m
        writes.append(("mlp", l, m))
        stages.append((f"mlp{l + 1}", x.clone()))
    logits = _ln(x, "ln_f") @ E.T
    return {"stages": stages, "acts": acts, "writes": writes, "logits": logits}


def neuron_push(l, n, k=5):
    """Roughly which tokens neuron n of layer l pushes toward: its output weights projected through the
    unembedding (centred and scaled by the final LayerNorm gain; ignores the LN's per-position 1/std)."""
    d = W[f"blocks.{l}.mlp_out.weight"][:, n]
    d = d - d.mean()
    proj = E @ (GAMMA * d)
    t = torch.topk(proj, k)
    return [(VOCAB[int(i)], float(s)) for s, i in zip(t.values, t.indices)], proj


_CORPUS_CACHE = {}


@torch.no_grad()
def corpus_chunks(n_docs=400, seed=7, T=64):
    """Fresh tiny-world stories (same generator as the training corpus, different seed), cut into
    T-token chunks. Returns (ids tensor (N, T), run() trace)."""
    key = (n_docs, seed, T)
    if key not in _CORPUS_CACHE:
        text = _corpus.make_corpus(n_docs, seed=seed)
        ids = tok(text)
        n = len(ids) // T
        X = torch.tensor(ids[: n * T]).view(n, T)
        _CORPUS_CACHE[key] = (X, run(X[:, : C["n_ctx"]]))   # T = 65 → 64 inputs, 64 shifted targets
    return _CORPUS_CACHE[key]


# --------------------------------------------------------------------------- loss
@torch.no_grad()
def token_losses(text):
    ids = tok(text)
    lp = forward(ids).log_softmax(-1)
    out = []
    for k in range(len(ids) - 1):
        p = math.exp(float(lp[k, ids[k + 1]]))
        out.append((VOCAB[ids[k]], VOCAB[ids[k + 1]], p, -math.log(p)))
    return out


def params_count():
    return sum(int(np.prod(t["shape"])) for t in META["tensors"])
