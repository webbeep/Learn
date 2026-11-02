"""Overfitting: same model, same number of steps, less and less training data.

python3 train/experiments/overfit.py [--steps 1500] [--workers 2]  →  public/research/overfit.json
"""
import argparse
from multiprocessing import Pool
from common import train_run, load_tokens, finite_last, dump, make_corpus, dedup_val

SIZES = [4096, 32768, 262144, None]  # None = all training tokens


def label(n, full):
    return f"all {full / 1e6:.1f}M tokens" if n is None else (f"{n // 1024}k tokens")


def one(args):
    n, steps, batch = args
    full = len(load_tokens()[0])
    r = train_run({}, steps=steps, batch=batch, train_data=n, threads=1, extra_evals={"val_dedup": dedup_val()[0]},
                  log=lambda s: print(n, s, flush=True))
    c = r["curve"]; nt = n or full
    vd = c["val_dedup"]
    return {"label": label(n, full), "data_tokens": nt, "epochs": round(steps * batch * 64 / nt, 2),
            "curve": {k: c[k] for k in ("step", "train", "val", "val_dedup")},
            "final_train": finite_last(c["train"]), "final_val": finite_last(c["val"]),
            "best_val": min(c["val"]), "best_val_step": c["step"][c["val"].index(min(c["val"]))],
            "final_val_dedup": finite_last(vd), "best_val_dedup": min(vd),
            "best_val_dedup_step": c["step"][vd.index(min(vd))]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        runs = pool.map(one, [(n, a.steps, a.batch) for n in SIZES], chunksize=1)
    tr, va = load_tokens()
    _, dd = dedup_val()
    dump("overfit.json", {
        "steps": a.steps, "batch": a.batch, "train_tokens": len(tr), "val_tokens": len(va),
        "corpus_chars": len(make_corpus()), "val_dedup": dd,
        "note_dedup": ("Leakage check: the tiny world has a finite grammar, so EVERY validation sentence also occurs in "
                       f"training, and {dd['val_lines_in_train']} of {dd['val_lines']} validation lines (whole stories) "
                       "appear verbatim in the training text. val_dedup = loss on 4 fixed batches of only the "
                       f"{dd['val_dedup_lines']} validation lines that never appear verbatim in training "
                       f"({dd['val_dedup_tokens']:,} tokens): held-out documents, i.e. new combinations of known templates, "
                       "not new kinds of sentences. Real corpora are deduplicated before splitting."),
        "note": (f"Baseline TinyGPT trained {a.steps} steps of {a.batch}×64 tokens on the first N training tokens. "
                 "train = loss on 4 fixed batches drawn from that same subset; val = loss on 4 fixed batches of the "
                 "shared held-out set (same for every run). AdamW lr 3e-3, warmup 100, cosine, wd 0.05, clip 1.0."),
        "runs": runs})
