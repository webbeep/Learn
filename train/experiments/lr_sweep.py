"""Learning-rate / optimizer sweep on the baseline TinyGPT.

Constant learning rate, no warmup, no gradient clipping (so "too big" is allowed to blow up).
python3 train/experiments/lr_sweep.py [--steps 1000] [--workers 2]  →  public/research/lr_sweep.json
"""
import argparse
from multiprocessing import Pool
from common import train_run, finite_last, dump

SWEEP = ([("adamw", lr) for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)] +
         [("sgd", lr) for lr in (0.1, 0.3, 1.0, 3.0)] +
         [("sgd-momentum", lr) for lr in (0.03, 0.1, 0.3, 1.0)])


def one(args):
    (opt, lr), steps, batch = args
    r = train_run({}, steps=steps, batch=batch, lr=lr, opt="adamw" if opt == "adamw" else "sgd",
                  momentum=0.9 if opt == "sgd-momentum" else 0.0, warmup=0, schedule="constant", clip=0,
                  eval_every=25, threads=1, log=lambda s: print(opt, lr, s, flush=True))
    c = r["curve"]; fv = None if c["val"][-1] is None else c["val"][-1]  # null when the run blew up (NaN)
    diverged = r["diverged"] or fv is None or fv >= c["val"][0]
    return {"optimizer": opt, "lr": lr, "momentum": 0.9 if opt == "sgd-momentum" else 0.0,
            "curve": {k: c[k] for k in ("step", "train", "val")}, "final_val": fv,
            "best_val": min(v for v in c["val"] if v is not None), "diverged": diverged}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        runs = pool.map(one, [(s, a.steps, a.batch) for s in SWEEP], chunksize=1)
    dump("lr_sweep.json", {
        "steps": a.steps, "batch": a.batch,
        "note": (f"Baseline TinyGPT (3 layers, d_model 64), same seed and data order for every run, {a.steps} steps of "
                 f"{a.batch}×64 tokens, CONSTANT learning rate (no warmup, no decay) and NO gradient clipping. "
                 "AdamW uses weight decay 0.05; SGD has no momentum; sgd-momentum uses momentum 0.9. "
                 "diverged = loss became NaN/inf or >20, or the final val loss is no better than at step 0."),
        "runs": runs})
