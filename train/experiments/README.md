# Research experiments

Small, real, CPU-only experiments on the TinyGPT family. Their JSON output in `public/research/` feeds the
labs (Architecture lab, Scaling lab, Train lab) and the "training" / "design" explainer videos.

Everything reuses `train/corpus.py` and `train/bpe.py`. The tokenizer is **not** retrained: vocab + merges are
loaded from `public/model/meta.json` and the corpus is encoded with `bpe.encode`, so token ids are identical to the
shipped browser model. Train/val split matches `train/train.py` (last 5% of 1,686,858 tokens = validation).
The encoded corpus is cached in `train/experiments/.cache/tokens.npy` (delete it to re-encode).

## Rerun

```bash
cd train/experiments
python3 ablations.py    # ~3 min  → public/research/ablations.json
python3 overfit.py      # ~2 min  → public/research/overfit.json
python3 lr_sweep.py     # ~4 min  → public/research/lr_sweep.json
python3 scaling.py      # ~15 min → public/research/scaling.json
python3 depth.py        # ~5 min  → public/research/depth.json
python3 gradients.py    # seconds → public/research/gradients.json
```

Each script runs its independent runs in a process pool (`--workers 2`, one torch thread each) and takes
`--steps` / `--batch` flags. Every run starts from seed 0 and sees the same batch order. Times are on a
shared 4-core CPU.

`common.py` holds the shared pieces:

* `TinyGPT(cfg)`: the model copied from `train/train.py` (pre-LN, GELU-tanh, learned positions, weight-tied
  unembedding, init std 0.02) with switches `n_layer, d_model, n_head, d_mlp, no_attention, no_mlp, no_positions,
  no_residual, no_layernorm`.
  * `no_residual`: `x = attn(ln1(x)); x = mlp(ln2(x))` instead of `x = x + …`.
  * `no_layernorm`: every LayerNorm (incl. the final one) becomes the identity.
  * `no_positions`: the position embedding is zeroed and frozen.
  * `no_attention` / `no_mlp`: the sublayer is skipped (its parameters are not counted).
* `train_run(...)`: AdamW (wd 0.05) or SGD, linear warmup + cosine or constant LR, optional grad clipping.
  Every `eval_every` steps it records **train** = mean loss on 4 fixed batches of the run's training data and **val** =
  mean loss on 4 fixed held-out batches (same for every run). Loss is cross-entropy in nats; ln 256 = 5.55 is chance.
* Samples: temperature 0.8, `torch.Generator` seed 1, 24 new tokens.

Default training recipe (unless a file's `note` says otherwise): batch 32 × 64 tokens (2,048 tokens/step),
AdamW lr 3e-3, 100 warmup steps, cosine decay to 0, weight decay 0.05, grad-norm clip 1.0.

## JSON schemas

All loss values are floats (4 decimals) or `null` once a run has diverged (curves stop at the divergence point).
Every file also has a free-text `note` describing the exact setup.

### `ablations.json`

```jsonc
{
  "steps": 1500, "batch": 32, "tokens_per_step": 2048, "note": "...",
  "probe_prompt": "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the",
  "runs": [{
    "id": "baseline" | "no-attention" | "no-mlp" | "no-positions" | "no-residual" | "no-layernorm" | "1-layer" | "6-layer",
    "label": "no attention",
    "what": "one sentence: what was removed and why it matters",
    "config": {"vocab_size": 256, "n_ctx": 64, "d_model": 64, "n_head": 4, "n_layer": 3, "d_mlp": 256,
               "no_attention": false, "no_mlp": false, "no_positions": false, "no_residual": false, "no_layernorm": false},
    "params": 170560,                        // trainable params actually used
    "curve": {"step": [0, 50, ...], "train": [...], "val": [...]},
    "final_val": 0.55, "final_train": 0.55, "diverged": false, "seconds": 60.1,
    "samples": [{"prompt": "...", "text": "continuation only (prompt not repeated)"}],  // same 3 prompts, same order:
              // probe prompt, "once upon a time", "q: what does the cat say? a:"
    "probe_top": [{"token": " red", "p": 0.87}, ...]   // top-5 next tokens for the probe prompt
  }]
}
```

### `scaling.json`

```jsonc
{
  "steps": 1500, "batch": 32, "lr": 0.003, "tokens_per_step": 2048, "note": "...",
  "runs": [{
    "id": "xs" | "s" | "m" | "base" | "l" | "xl" | "xxl", "d_model": 16, "n_layer": 1, "n_head": 2, "d_mlp": 64,
    "params": 9000, "non_embedding_params": 3300,
    "tokens_seen": 3072000, "flops": 1.6e11,           // 6 · params · tokens_seen
    "curve": {"step": [...], "tokens": [...], "flops": [...], "train": [...], "val": [...]},
    "final_val": 1.2, "final_train": 1.2, "seconds": 30.0
  }],
  "fit": {"formula": "L(N) = c + a * N^(-alpha)", "a": 0.0, "alpha": 0.0, "c": 0.0,
          "fit_on": "non_embedding_params", "r2": 0.99, "pred": [...]}   // pred = fitted L for each run, same order
}
```

### `lr_sweep.json`

Constant LR, no warmup, no clipping, 1000 steps, eval every 25 steps.

```jsonc
{
  "steps": 1000, "batch": 32, "note": "...",
  "runs": [{"optimizer": "adamw" | "sgd" | "sgd-momentum", "lr": 0.003, "momentum": 0.9,
            "curve": {"step": [...], "train": [...], "val": [...]},
            "final_val": 0.7, "best_val": 0.7, "diverged": false}]
}
```

`diverged` = loss became NaN/inf or > 20, or the final val loss is no better than at step 0.

### `overfit.json`

```jsonc
{
  "steps": 1500, "batch": 32, "note": "...",
  "runs": [{"label": "4k tokens", "data_tokens": 4096, "epochs": 750.0,
            "curve": {"step": [...], "train": [...], "val": [...]},
            "final_train": 0.1, "final_val": 2.0, "best_val": 1.5, "best_val_step": 300}]
}
```

### `gradients.json`

Gradients of the shipped model (`public/model/weights.bin`) and of a fresh init (seed 0) on one fixed training batch
(32 × 64 tokens), no clipping.

```jsonc
{
  "note": "...", "batch": {"size": 32, "n_ctx": 64, "loss_trained": 0.46, "loss_step0": 5.56},
  "per_tensor": [{"name": "blocks.0.attn_qkv.weight", "group": "block 1", "shape": [192, 64],
                  "grad_norm": 0.01, "param_norm": 8.0, "grad_norm_step0": 0.1, "param_norm_step0": 2.2,
                  "numel": 12288, "grad_rms": 0.0006, "grad_rms_step0": 0.0003}],   // rms = norm / sqrt(numel)
  "per_layer_norm_at": {"layers": ["embeddings", "block 1", "block 2", "block 3", "final LayerNorm"],
                        "step0": [...], "trained": [...]},       // L2 norm of all grads in that group
  "example": {"prompt": "...now tom has the", "target": " red", "p_target": 0.871, "loss": 0.138,
              "dloss_dlogit_top": [{"token": " red", "grad": -0.129, "p": 0.871}, ...],   // top 8 by |grad|
              "explain": "..."}
}
```

### `depth.json`

Why deep stacks need residuals and LayerNorm.

```jsonc
{
  "note": "...", "init_depth": 24, "train_depth": 12, "steps": 1000,
  "init": [{"id": "baseline" | "no-residual" | "no-layernorm" | "neither", "label": "...",
            "act_rms": [...],          // RMS of the hidden state after layer i (index 0 = embeddings), at init
            "grad_rms": [...],         // RMS of dLoss/d(hidden state after layer i), at init
            "block_grad_norm": [...],  // L2 norm of each block's weight gradient, at init
            "loss": 5.55}],
  "runs": [{"id": "...", "label": "...", "config": {...}, "params": 600000,
            "curve": {"step": [...], "train": [...], "val": [...]}, "final_val": 0.6, "diverged": false,
            "act_rms_trained": [...]}]   // RMS of the hidden state after each layer of the TRAINED model
}
```
