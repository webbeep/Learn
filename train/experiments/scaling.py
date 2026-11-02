"""Scaling: train models from ~10k to ~1.8M parameters on the same token budget, fit L(N) = c + a·N^-alpha.

python3 train/experiments/scaling.py [--steps 1500] [--workers 2]  →  public/research/scaling.json
"""
import argparse
from multiprocessing import Pool
import numpy as np
from scipy.optimize import curve_fit
from common import train_run, finite_last, dump

SIZES = [  # id, d_model, n_layer, n_head
    ("xs", 16, 1, 2), ("s", 24, 2, 2), ("m", 32, 2, 4), ("base", 64, 3, 4),
    ("l", 96, 4, 4), ("xl", 128, 4, 4), ("xxl", 192, 4, 6),
]


def one(args):
    (rid, d, L, h), steps, batch, lr = args
    cfg = dict(d_model=d, n_layer=L, n_head=h, d_mlp=4 * d)
    r = train_run(cfg, steps=steps, batch=batch, lr=lr, threads=1, log=lambda s: print(rid, s, flush=True))
    m, c = r["model"], r["curve"]
    N, Nne = m.n_params(), m.n_non_embedding()
    return {"id": rid, "d_model": d, "n_layer": L, "n_head": h, "d_mlp": 4 * d, "params": N,
            "non_embedding_params": Nne, "tokens_seen": c["tokens"][-1], "flops": 6 * N * c["tokens"][-1],
            "curve": {"step": c["step"], "tokens": c["tokens"], "flops": [6 * N * t for t in c["tokens"]],
                      "train": c["train"], "val": c["val"]},
            "final_val": finite_last(c["val"]), "final_train": finite_last(c["train"]), "seconds": r["seconds"]}


def law(N, a, alpha, c): return c + a * N ** (-alpha)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        runs = pool.map(one, [(s, a.steps, a.batch, a.lr) for s in reversed(SIZES)], chunksize=1)[::-1]
    N = np.array([r["non_embedding_params"] for r in runs], float); L = np.array([r["final_val"] for r in runs])
    (pa, palpha, pc), _ = curve_fit(law, N, L, p0=(10.0, 0.3, 0.3), bounds=([0, 0, 0], [1e4, 2, min(L)]), maxfev=20000)
    pred = law(N, pa, palpha, pc)
    r2 = 1 - ((L - pred) ** 2).sum() / ((L - L.mean()) ** 2).sum()
    dump("scaling.json", {
        "steps": a.steps, "batch": a.batch, "lr": a.lr, "tokens_per_step": a.batch * 64,
        "note": (f"Every size trained on the same {a.steps * a.batch * 64:,} tokens ({a.steps} steps of {a.batch}×64), "
                 f"AdamW lr {a.lr:g}, 100 warmup steps, cosine decay, weight decay 0.05, grad clip 1.0, seed 0. d_mlp = 4·d_model. "
                 "flops ≈ 6 · params · tokens (all parameters incl. the tied unembedding; attention-score FLOPs ignored). "
                 "val = mean cross-entropy (nats) on 4 fixed held-out batches."),
        "runs": runs,
        "fit": {"formula": "L(N) = c + a * N^(-alpha)", "a": float(pa), "alpha": float(palpha), "c": float(pc),
                "fit_on": "non_embedding_params", "r2": float(r2),
                "pred": [round(float(p), 4) for p in pred]}})
