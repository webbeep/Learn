"""Follow-up to depth.py (adds keys to public/research/depth.json, keeps everything else):

* init[*].token_spread: at initialisation, how different the token positions still are after each layer
  (std across positions / RMS). Without residuals all positions collapse to the same vector.
* shallow: the 3-layer model at init (gradient RMS per layer) and 600-step runs of a 3-layer NO-residual model with
  the standard init (std 0.02) vs a larger block init (std 0.125 ≈ 1/sqrt(d_model)) - how stuck a plain stack gets
  depends on the init.

python3 train/experiments/nores_init.py
"""
import json, os
from multiprocessing import Pool
import torch
import torch.nn.functional as F
from common import OUT, TinyGPT, load_tokens, fixed_batches, train_run, finite_last, dump
from depth import VARIANTS


def spread_and_grads(over, depth, x, y):
    torch.manual_seed(0); m = TinyGPT({"n_layer": depth, **over})
    h = m.tok(x) + m.pos(torch.arange(x.shape[1])); hs = [h]; h.retain_grad()
    for b in m.blocks:
        h = b(h); h.retain_grad(); hs.append(h)
    F.cross_entropy((m.ln_f(h) @ m.tok.weight.T).view(-1, m.c["vocab_size"]), y.view(-1)).backward()
    spread = [float(a.detach().std(dim=1).mean() / a.detach().pow(2).mean().sqrt()) for a in hs]
    return spread, [float(a.grad.pow(2).mean().sqrt()) for a in hs], [float(a.detach().pow(2).mean().sqrt()) for a in hs]


def run(std):
    r = train_run({"no_residual": True, "init_std": std}, steps=600, batch=32, threads=1, eval_every=50,
                  log=lambda s: print("std", std, s, flush=True))
    c = r["curve"]
    return {"init_std": std, "curve": {k: c[k] for k in ("step", "train", "val")}, "final_val": finite_last(c["val"])}


if __name__ == "__main__":
    torch.set_num_threads(2)
    path = os.path.join(OUT, "depth.json"); d = json.load(open(path))
    x, y = fixed_batches(load_tokens()[0], 1, 16, 64, 99)[0]
    for e, (rid, _, over) in zip(d["init"], VARIANTS):
        assert e["id"] == rid
        e["token_spread"] = spread_and_grads(over, d["init_depth"], x, y)[0]
    shallow = {}
    for rid, lab, over in VARIANTS[:2]:
        sp, g, a = spread_and_grads(over, 3, x, y)
        shallow[rid] = {"grad_rms": g, "act_rms": a, "token_spread": sp}
    with Pool(2) as pool:
        runs = pool.map(run, [0.02, 0.125])
    d["shallow"] = {"n_layer": 3, "init": shallow, "nores_runs": runs,
                    "note": ("3-layer TinyGPT at init (same batch as init[*]): gradient / activation RMS and token "
                             "spread per layer, with and without residuals. nores_runs: 3-layer NO-residual model "
                             "trained 600 steps of 32x64 tokens (AdamW 3e-3, warmup 100, cosine, clip 1.0) with block "
                             "weights initialised at std 0.02 (GPT-2 default) vs 0.125 (~1/sqrt(64)).")}
    d["note_token_spread"] = ("token_spread[i] = mean over features of the std across the 64 positions of the hidden "
                              "state after layer i, divided by its RMS: ~0 means every position holds the same vector.")
    dump("depth.json", d)
