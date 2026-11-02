"""Overfitting: same model, same number of steps, less and less training data.

python3 train/experiments/overfit.py [--steps 1500] [--workers 2]  →  public/research/overfit.json
"""
import argparse
from multiprocessing import Pool
from common import train_run, load_tokens, finite_last, dump, make_corpus

SIZES = [4096, 32768, 262144, None]  # None = all training tokens


def label(n, full):
    return f"all {full / 1e6:.1f}M tokens" if n is None else (f"{n // 1024}k tokens")


def one(args):
    n, steps, batch = args
    full = len(load_tokens()[0])
    r = train_run({}, steps=steps, batch=batch, train_data=n, threads=1,
                  log=lambda s: print(n, s, flush=True))
    c = r["curve"]; nt = n or full
    return {"label": label(n, full), "data_tokens": nt, "epochs": round(steps * batch * 64 / nt, 2),
            "curve": {k: c[k] for k in ("step", "train", "val")},
            "final_train": finite_last(c["train"]), "final_val": finite_last(c["val"]),
            "best_val": min(c["val"]), "best_val_step": c["step"][c["val"].index(min(c["val"]))]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        runs = pool.map(one, [(n, a.steps, a.batch) for n in SIZES], chunksize=1)
    tr, va = load_tokens()
    dump("overfit.json", {
        "steps": a.steps, "batch": a.batch, "train_tokens": len(tr), "val_tokens": len(va),
        "corpus_chars": len(make_corpus()),
        "note": (f"Baseline TinyGPT trained {a.steps} steps of {a.batch}×64 tokens on the first N training tokens. "
                 "train = loss on 4 fixed batches drawn from that same subset; val = loss on 4 fixed batches of the "
                 "shared held-out set (same for every run). AdamW lr 3e-3, warmup 100, cosine, wd 0.05, clip 1.0."),
        "runs": runs})
