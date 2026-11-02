"""Why depth needs residuals + LayerNorm: signal and gradient size through a deep stack at initialisation,
plus short training runs of a 12-layer TinyGPT with and without residuals / LayerNorm.

python3 train/experiments/depth.py [--steps 1000]  →  public/research/depth.json
"""
import argparse
from multiprocessing import Pool
import torch
import torch.nn.functional as F
from common import TinyGPT, load_tokens, fixed_batches, train_run, finite_last, dump

VARIANTS = [("baseline", "residual + LayerNorm", {}), ("no-residual", "no residual", {"no_residual": True}),
            ("no-layernorm", "no LayerNorm", {"no_layernorm": True}),
            ("neither", "neither", {"no_residual": True, "no_layernorm": True})]


def probe_init(over, depth, x, y):
    torch.manual_seed(0); m = TinyGPT({"n_layer": depth, **over})
    acts = []
    h = m.tok(x) + m.pos(torch.arange(x.shape[1])); h.retain_grad(); acts.append(h)
    for b in m.blocks:
        h = b(h); h.retain_grad(); acts.append(h)
    loss = F.cross_entropy((m.ln_f(h) @ m.tok.weight.T).view(-1, m.c["vocab_size"]), y.view(-1)); loss.backward()
    return {"act_rms": [float(a.detach().pow(2).mean().sqrt()) for a in acts],
            "grad_rms": [float(a.grad.pow(2).mean().sqrt()) for a in acts],
            "block_grad_norm": [float(torch.sqrt(sum((p.grad ** 2).sum() for p in b.parameters()))) for b in m.blocks],
            "loss": float(loss.detach())}


@torch.no_grad()
def act_profile(m, x):
    m.eval(); h = m.tok(x) + m.pos(torch.arange(x.shape[1])); out = [float(h.pow(2).mean().sqrt())]
    for b in m.blocks:
        h = b(h); out.append(float(h.pow(2).mean().sqrt()))
    return out


def one(args):
    (rid, label, over), steps, depth = args
    r = train_run({"n_layer": depth, **over}, steps=steps, batch=32, threads=1, log=lambda s: print(rid, s, flush=True))
    c = r["curve"]
    x, _ = fixed_batches(load_tokens()[0], 1, 16, 64, 99)[0]
    return {"id": rid, "label": label, "config": {"n_layer": depth, **over}, "params": r["model"].n_params(),
            "curve": {k: c[k] for k in ("step", "train", "val")}, "final_val": finite_last(c["val"]),
            "diverged": r["diverged"],
            "act_rms_trained": act_profile(r["model"], x)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=12); ap.add_argument("--init_depth", type=int, default=24)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(2)
    x, y = fixed_batches(load_tokens()[0], 1, 16, 64, 99)[0]
    init = [{"id": rid, "label": lab, **probe_init(o, a.init_depth, x, y)} for rid, lab, o in VARIANTS]
    with Pool(a.workers) as pool:
        runs = pool.map(one, [(v, a.steps, a.depth) for v in VARIANTS], chunksize=1)
    dump("depth.json", {
        "note": (f"init: one batch (16×64 tokens) through a freshly initialised {a.init_depth}-layer TinyGPT (d_model 64, "
                 "weights std 0.02, seed 0). act_rms[i] / grad_rms[i] = RMS of the residual-stream activation after layer i "
                 "(0 = embeddings) and of dLoss/d(that activation). block_grad_norm = L2 norm of each block's weight gradient. "
                 f"runs: {a.depth}-layer models trained {a.steps} steps of 32×64 tokens, AdamW 3e-3, warmup 100, cosine, clip 1.0."),
        "init_depth": a.init_depth, "init": init, "train_depth": a.depth, "steps": a.steps, "runs": runs})
