"""Real mini-experiment for the 'research-loop' video: does TinyGPT need positional embeddings?

Two short CPU runs (2 threads), identical in everything (code from train/train.py, seed, batches, lr schedule,
steps) except ONE change: the learned position embedding is removed from the input sum.
Writes video/data/research_nopos.json, which the scene reads at render time.

    python3 video/exp_research.py              # the two runs (~4 min each on 2 threads)
    python3 video/exp_research.py --released   # + the published model on the same eval batches
    python3 video/exp_research.py --leak       # recompute the train/val overlap check only
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_concepts import TinyGPT, corpus_tokens, DATA, PROBE, META, VOCAB, tok  # noqa: E402

torch.set_num_threads(2)
STEPS = int(os.environ.get("EXP_STEPS", 600))
BATCH, LR, SEED, EVAL_EVERY, N_EVAL = 64, 3e-3, 0, 50, 24
CFG = dict(META["config"])

data = torch.tensor(corpus_tokens().astype(np.int64))
n_val = len(data) // 20
train_d, val_d = data[:-n_val], data[-n_val:]
T = CFG["n_ctx"]

# the SAME batches for both runs, and a FIXED held-out evaluation set
g = torch.Generator().manual_seed(1234)
train_ix = torch.randint(len(train_d) - T - 1, (STEPS + 1, BATCH), generator=g)
val_ix = torch.randint(len(val_d) - T - 1, (N_EVAL, BATCH), generator=g)


def get(d, ix):
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + 1 + T] for i in ix])


@torch.no_grad()
def val_loss(m):
    m.eval()
    ls = [F.cross_entropy(m(x).view(-1, CFG["vocab_size"]), y.view(-1)).item() for x, y in (get(val_d, ix) for ix in val_ix)]
    m.train()
    return float(np.mean(ls)), float(np.std(ls) / math.sqrt(len(ls)))


@torch.no_grad()
def probe(m, prompt, cands):
    m.eval()
    p = m(torch.tensor([tok(prompt)]))[0, -1].softmax(-1)
    top = torch.argsort(p, descending=True)[:3].tolist()
    m.train()
    return {"top": [[VOCAB[t], float(p[t])] for t in top], **{c: float(p[VOCAB.index(c)]) for c in cands}}


def run(use_pos):
    torch.manual_seed(SEED)
    m = TinyGPT(CFG, use_pos=use_pos)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.05)
    log = {"steps": [], "train": [], "val": [], "val_se": []}
    t0 = time.time()
    for step in range(STEPS + 1):
        lr = LR * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for g_ in opt.param_groups: g_["lr"] = lr
        x, y = get(train_d, train_ix[step])
        loss = F.cross_entropy(m(x).view(-1, CFG["vocab_size"]), y.view(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % EVAL_EVERY == 0:
            vl, se = val_loss(m)
            log["steps"].append(step); log["train"].append(round(loss.item(), 4))
            log["val"].append(round(vl, 4)); log["val_se"].append(round(se, 4))
            print(f"pos={use_pos} step {step} train {loss.item():.3f} val {vl:.3f}±{se:.3f} {time.time() - t0:.0f}s", flush=True)
    log["seconds"] = round(time.time() - t0, 1)
    log["params"] = sum(p.numel() for n, p in m.named_parameters() if use_pos or n != "pos.weight")
    log["probe"] = probe(m, PROBE, [" red", " blue"])
    return log


def leakage():
    """How many held-out documents (lines of the validation split) also appear word for word in the training split?"""
    val_text = "".join(VOCAB[t] for t in val_d.tolist())
    train_text = "".join(VOCAB[t] for t in train_d.tolist())
    val_docs = val_text.split("\n")[1:-1]          # drop the partial first line and the trailing empty one
    train_docs = set(train_text.split("\n"))
    dup = sum(d in train_docs for d in val_docs)

    def kind(d):  # the corpus generators (train/corpus.py) leave clear fingerprints
        return "trade" if " gives the " in d else ("story" if d.startswith("once upon") else
                                                    ("trip" if " went to the " in d else "short"))
    by = {}
    for d in val_docs:
        k = kind(d); n, m = by.get(k, (0, 0)); by[k] = (n + 1, m + (d in train_docs))
    return dict(val_docs=len(val_docs), in_train=dup, frac=dup / len(val_docs), train_docs=len(train_docs),
                example=next(d for d in val_docs if d in train_docs),
                by_kind={k: {"n": n, "in_train": m} for k, (n, m) in by.items()})


if __name__ == "__main__" and "--leak" in sys.argv:
    path = os.path.join(DATA, "research_nopos.json")
    out = json.load(open(path)); out["leak"] = leakage()
    json.dump(out, open(path, "w"), indent=1)
    print(out["leak"])
elif __name__ == "__main__" and "--released" in sys.argv:
    # evaluate the published 3000-step model on the SAME fixed held-out batches (no training)
    from lib_concepts import load_tiny
    path = os.path.join(DATA, "research_nopos.json")
    out = json.load(open(path))
    vl, se = val_loss(load_tiny())
    out["released"] = {"steps": 3000, "val": round(vl, 4), "val_se": round(se, 4)}
    json.dump(out, open(path, "w"), indent=1)
    print(out["released"])
elif __name__ == "__main__":
    out = {"steps": STEPS, "batch": BATCH, "lr": LR, "seed": SEED, "threads": 2, "n_eval_batches": N_EVAL,
           "eval_tokens": N_EVAL * BATCH * T}
    out["baseline"] = run(True)
    out["nopos"] = run(False)
    out["leak"] = leakage()
    os.makedirs(DATA, exist_ok=True)
    json.dump(out, open(os.path.join(DATA, "research_nopos.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, list)})
                      for k, v in out.items()}, indent=1))
