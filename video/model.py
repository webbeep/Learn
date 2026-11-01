"""Tiny reference forward pass of TinyGPT (reads public/model/*), used to pull REAL numbers for the videos."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "train"))
from bpe import encode, pretokenize  # noqa: E402

META = json.load(open(os.path.join(ROOT, "public/model/meta.json")))
TRAIN = json.load(open(os.path.join(ROOT, "public/model/training.json")))
_w = np.fromfile(os.path.join(ROOT, "public/model/weights.bin"), dtype="<f4")
W = {t["name"]: torch.tensor(_w[t["offset"]:t["offset"] + int(np.prod(t["shape"]))].reshape(t["shape"]))
     for t in META["tensors"]}
C = META["config"]; D = C["d_model"]; H = C["n_head"]; HD = D // H
VOCAB = META["vocab"]; MERGES = META["merges"]


def tok(text):
    return encode(text, VOCAB, MERGES)


def _ln(x, n):
    return F.layer_norm(x, (D,), W[n + ".weight"], W[n + ".bias"])


def _lin(x, n):
    return x @ W[n + ".weight"].T + W[n + ".bias"]


@torch.no_grad()
def forward(ids, trace=False):
    T = len(ids)
    x = W["tok.weight"][ids] + W["pos.weight"][:T]
    tr = {"embed": x.clone(), "attn": [], "scores": [], "resid": []}
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for l in range(C["n_layer"]):
        p = f"blocks.{l}."
        q, k, v = _lin(_ln(x, p + "ln1"), p + "attn_qkv").split(D, -1)
        q, k, v = (t.view(T, H, HD).transpose(0, 1) for t in (q, k, v))
        s = q @ k.transpose(-1, -2) / HD ** 0.5
        s = s.masked_fill(mask, float("-inf"))
        a = s.softmax(-1)
        o = (a @ v).transpose(0, 1).reshape(T, D)
        x = x + _lin(o, p + "attn_out")
        tr["resid"].append(("attn", l, x.clone()))
        x = x + _lin(F.gelu(_lin(_ln(x, p + "ln2"), p + "mlp_in"), approximate="tanh"), p + "mlp_out")
        tr["resid"].append(("mlp", l, x.clone()))
        tr["attn"].append(a); tr["scores"].append(s)
    logits = _ln(x, "ln_f") @ W["tok.weight"].T
    return (logits, tr) if trace else logits


def lens(x):
    return _ln(x, "ln_f") @ W["tok.weight"].T


def bpe_steps(word):
    """Replay BPE encode for one pre-token, returning every merge applied."""
    rank = {(a, b): i for i, (a, b) in enumerate(MERGES)}
    s = list(word); steps = [list(s)]; applied = []
    while len(s) > 1:
        best = min(((rank.get((s[i], s[i + 1]), 1e9), i) for i in range(len(s) - 1)))
        if best[0] == 1e9:
            break
        i = best[1]
        applied.append({"rank": int(best[0]), "pair": [s[i], s[i + 1]], "pos": i, "new": s[i] + s[i + 1],
                        "id": VOCAB.index(s[i] + s[i + 1])})
        s = s[:i] + [s[i] + s[i + 1]] + s[i + 2:]
        steps.append(list(s))
    return steps, applied
