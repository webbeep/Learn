"""Ablations: remove one part of TinyGPT at a time and train each variant the same way.

python3 train/experiments/ablations.py [--steps 3000 --batch 64] [--workers 2]  →  public/research/ablations.json
"""
import argparse
from multiprocessing import Pool
from common import BASE_CFG, PROBE, PROMPTS, train_run, sample, top_next, finite_last, dump

RUNS = [
    ("baseline", "baseline", {}, "The full model: 3 layers of attention + MLP with residuals, LayerNorm and positions."),
    ("no-attention", "no attention", {"no_attention": True},
     "Attention removed, so each token only sees itself and cannot copy information from earlier words."),
    ("no-mlp", "no MLP", {"no_mlp": True},
     "MLP blocks removed, so the model can still mix tokens but has far less per-token processing and memory."),
    ("no-positions", "no positions", {"no_positions": True},
     "Positional embeddings removed, so the model knows which words came before but not in what order."),
    ("no-residual", "no residual", {"no_residual": True},
     "Residual connections removed, so each layer must rebuild the whole signal instead of adding to it."),
    ("no-layernorm", "no LayerNorm", {"no_layernorm": True},
     "LayerNorm removed, so activation sizes are no longer re-scaled and can drift from layer to layer."),
    ("1-layer", "1 layer", {"n_layer": 1}, "Only one block instead of three: less depth to build multi-step patterns."),
    ("6-layer", "6 layers", {"n_layer": 6}, "Six blocks instead of three: twice the depth and roughly 1.9× the parameters."),
]


def one(args):
    (rid, label, over, what), steps, batch = args
    r = train_run(over, steps=steps, batch=batch, threads=1, log=lambda s: print(rid, s, flush=True))
    m = r["model"]
    return {"id": rid, "label": label, "what": what, "config": {**BASE_CFG, **over}, "params": m.n_params(),
            "curve": {k: r["curve"][k] for k in ("step", "train", "val")}, "final_val": finite_last(r["curve"]["val"]),
            "final_train": finite_last(r["curve"]["train"]), "diverged": r["diverged"], "seconds": r["seconds"],
            "samples": [{"prompt": p, "text": sample(m, p)[len(p):]} for p in PROMPTS],
            "probe_top": top_next(m, PROBE, 5)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        runs = pool.map(one, [(r, a.steps, a.batch) for r in RUNS], chunksize=1)
    dump("ablations.json", {
        "steps": a.steps, "batch": a.batch, "tokens_per_step": a.batch * 64,
        "note": (f"Each variant trained from the same seed for {a.steps} steps of {a.batch}×64 tokens (the same recipe as the "
                 "shipped model) "
                 "(AdamW lr 3e-3, 100 warmup steps, cosine decay, weight decay 0.05, grad clip 1.0). Loss is "
                 "cross-entropy in nats on 4 fixed batches; samples use temperature 0.8, seed 1, 24 tokens."),
        "probe_prompt": PROBE, "runs": runs})
