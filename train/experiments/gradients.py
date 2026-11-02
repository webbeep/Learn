"""Real gradients of the shipped TinyGPT (public/model/weights.bin) on one fixed training batch.

python3 train/experiments/gradients.py  →  public/research/gradients.json
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT, META, VOCAB, PROBE, TinyGPT, load_tokens, fixed_batches, tok_ids, dump


def load_trained():
    m = TinyGPT(dict(META["config"]))
    flat = np.fromfile(os.path.join(ROOT, "public", "model", "weights.bin"), dtype="<f4")
    sd = {}
    for t in META["tensors"]:
        n = int(np.prod(t["shape"]))
        sd[t["name"]] = torch.from_numpy(flat[t["offset"]:t["offset"] + n].reshape(t["shape"]).copy())
    m.load_state_dict(sd); return m


def group(name):
    if name.startswith(("tok.", "pos.")): return "embeddings"
    if name.startswith("blocks."): return f"block {int(name.split('.')[1]) + 1}"
    return "final LayerNorm"


def grads(m, x, y):
    m.zero_grad(); m.train()
    loss = F.cross_entropy(m(x).view(-1, m.c["vocab_size"]), y.view(-1)); loss.backward()
    return loss.item(), {n: p.grad.detach().clone() for n, p in m.named_parameters()}


if __name__ == "__main__":
    torch.set_num_threads(2); torch.manual_seed(0)
    train_d, _ = load_tokens()
    x, y = fixed_batches(train_d, 1, 32, 64, 777)[0]
    trained = load_trained()
    torch.manual_seed(0); fresh = TinyGPT(dict(META["config"]))
    l_tr, g_tr = grads(trained, x, y); l_0, g_0 = grads(fresh, x, y)
    params = dict(trained.named_parameters()); params0 = dict(fresh.named_parameters())
    per_tensor = [{"name": n, "group": group(n), "shape": list(p.shape), "grad_norm": float(g_tr[n].norm()),
                   "param_norm": float(p.detach().norm()), "grad_norm_step0": float(g_0[n].norm()),
                   "param_norm_step0": float(params0[n].detach().norm()), "numel": p.numel(),
                   "grad_rms": float(g_tr[n].norm()) / p.numel() ** 0.5,
                   "grad_rms_step0": float(g_0[n].norm()) / p.numel() ** 0.5} for n, p in params.items()]
    layers = ["embeddings", "block 1", "block 2", "block 3", "final LayerNorm"]

    def per_layer(g):
        return [float(torch.sqrt(sum((g[n] ** 2).sum() for n in g if group(n) == L))) for L in layers]

    # one example: the probe prompt, next token " red"
    ids = tok_ids(PROBE); target = VOCAB.index(" red")
    trained.eval(); logits = trained(torch.tensor([ids]))[0, -1].detach().requires_grad_(True)
    loss = F.cross_entropy(logits[None], torch.tensor([target])); loss.backward()
    p = F.softmax(logits, -1).detach(); g = logits.grad
    order = g.abs().argsort(descending=True)[:8]
    dump("gradients.json", {
        "note": ("Gradients of mean cross-entropy w.r.t. every weight tensor on one fixed training batch of 32×64 tokens, "
                 "for the shipped trained TinyGPT and for a freshly initialised one (seed 0, std 0.02). No clipping."),
        "batch": {"size": 32, "n_ctx": 64, "loss_trained": l_tr, "loss_step0": l_0},
        "per_tensor": per_tensor,
        "per_layer_norm_at": {"layers": layers, "step0": per_layer(g_0), "trained": per_layer(g_tr)},
        "example": {"prompt": PROBE, "target": " red", "p_target": float(p[target]), "loss": float(loss.detach()),
                    "dloss_dlogit_top": [{"token": VOCAB[int(i)], "grad": float(g[i]), "p": float(p[i])} for i in order],
                    "explain": "dLoss/dlogit = softmax probability − 1 for the target, = probability for every other token."}})
    print("p(red) =", float(p[target]), "loss", float(loss.detach()), "batch loss trained", l_tr, "step0", l_0)
