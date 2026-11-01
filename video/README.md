# Explainer videos

Five narrated explainers (3Blue1Brown-style visuals in Manim, Fireship pacing) built from the **real** TinyGPT
in `public/model/`. Every number on screen is computed from `weights.bin` / `meta.json` / `training.json` at
render time (`model.py` is a small torch re-implementation of the forward pass).

| id | file | about |
|---|---|---|
| `gpt-in-100s` | `scenes/v1_gpt.py` | whole pipeline on "…now tom has the" → " red" (87%) |
| `tokens` | `scenes/v2_tokens.py` | BPE: base symbols, pair counts, merges replayed on " wednesday" |
| `attention` | `scenes/v3_attention.py` | q/k/v, dot product, softmax, causal mask, real L2H4 color-copy head |
| `next-token` | `scenes/v4_next.py` | logits → softmax, temperature sweep, top-k, weighted dice, a real mistake |
| `training` | `scenes/v5_training.py` | init, cross-entropy, gradient descent, real loss curve + samples |

## Build

```bash
# needs: manim 0.22 (no LaTeX required), ffmpeg/ffprobe, torch, fonts "Inter" + "JetBrains Mono"
export CLOUDFLARE_ACCOUNT_ID=...  CLOUDFLARE_API_TOKEN=...   # narration (Workers AI, Deepgram Aura-2)
python video/build.py               # render all → public/videos/<id>.mp4 + <id>.jpg + index.json
python video/build.py tokens        # just one (index.json is rebuilt from every built video)
python video/build.py --preview     # quick 480p15 renders into video/media/ (nothing published)
python video/build.py --index-only  # regenerate public/videos/index.json from video/build/*.json
```

Rendering is 1920×1080 @ 30 fps; `build.py` re-encodes with x264 (crf 24, raising it automatically if a file
would exceed ~24 MB, the Cloudflare Pages per-file limit), AAC 128k, loudness-normalised, `+faststart`, and
grabs a 1280×720 poster (< 200 KB).

Narration audio is cached in `video/.cache/` keyed by a hash of (voice, text), so re-renders never re-request
TTS. Change a line of narration and only that line is regenerated. The voice is `orion` (set `VIDEO_VOICE` to
override).

## Framework (`vscene.py`)

* `VScene` — dark background `#0b0e17`, palette (`BLUE TEAL YELLOW ORANGE PINK MUTED FG`), `Inter` / `JetBrains Mono`.
* `with self.say(text, scene=..., chapter=...) as line:` — adds the narration audio at the current time, lets
  you animate inside the block, then waits out the rest of the audio. `self.sync("phrase")` waits until roughly
  when that phrase is spoken (character-proportional estimate). Records a segment `{start, end, text, scene, objects}`.
* `self.mark(id, mobject, label)` — registers a clickable object. At the end of every `say` block each marked
  mobject that is on screen (and not fully transparent) is snapshotted with a normalised `[x, y, w, h]` box
  (top-left origin). `self.unmark(...)` drops ids.
* `self.chip(token, color)` — token chip (leading space shown as `␣`), `self.flash(text)` — punchline pop.
* On tear-down the timeline goes to `video/build/<id>.json`; `build.py` merges these into
  `public/videos/index.json` (the web player's contract: id, title, subtitle, duration, src, poster, relates,
  voice, chapters, segments).

`checkframes.py <mp4> <build json>` extracts a frame per segment into `video/frames/<id>/` with the object
boxes drawn on top; `sheet.py <dir>` tiles them into a contact sheet for quick visual QA.
