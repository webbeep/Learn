# 🔭 Top-Down AI

Learn how AI models work **top-down**: start with a real model running in front of you, poke it, and dive into
the fundamentals only when you hit a question. Every visual is clickable, and an AI tutor answers questions about
exactly what you selected.

**Live:** https://topdownai.pages.dev

## What's inside

- **Watch** (home): 18 short videos, played back to back. Pause and tap anything to ask about it.
- **Learn**: the same 18 topics as chapters, each with key ideas, a quick check, and optional math + code.
- **Labs**: X-ray, Train a model, Backprop, Training replay, Remove parts, Scaling, Your experiment.
- **Ask**: a tutor that sees what you're looking at. Type or talk.

## How "ask about anything" works

- Every visual element has `data-ask-id` and `data-ask` (a plain-English description containing the real numbers).
- Clicking, shift-clicking or region-selecting gathers those into the **selection**. Each page also publishes a summary of what's on screen.
- `functions/api/ask.ts` (a Cloudflare Pages Function) sends the selection, the on-screen elements and your learner profile to the LLM and streams the answer back.
- The tutor can **point** at elements with `[[element-id]]` (hover the ◎ chip to highlight the element) and suggest **experiments** with `[[try: prompt]]`, which you can click to run.
- `functions/api/profile.ts` rebuilds your learner profile every few questions and adds ideas to the build list.

## Talk to it

- **🎙 Voice in:** press <kbd>V</kbd> and talk (it sends when you go quiet), or hold <kbd>V</kbd> for push-to-talk. You can also use the mic button.
  It uses the browser's live speech recognition, or **Whisper on Workers AI** (`/api/transcribe`), which is more accurate and works in every browser.
- **🔊 Voice out:** spoken answers use a natural **Deepgram Aura** voice from Workers AI (`/api/speak`) or the browser's voices.
  Sentences are spoken while the answer is still streaming. When you speak, the tutor switches to a conversational style.
- **Hands-free mode:** after each spoken answer it starts listening again.
- **When you highlight** text, or drag a box with <kbd>H</kbd>: show options (🎙 Talk / ⌨️ Type / ✨ Explain), start listening, or explain right away. Pick one under ⚙.
- **It knows what you're looking at:** it receives only the elements visible in the window, your selection or highlighted text,
  and what your mouse is over (so "what's *this*?" works). 👁 in the panel shows exactly what it receives.
- Shortcuts: <kbd>V</kbd> talk · <kbd>H</kbd> highlight · <kbd>E</kbd> explain · <kbd>/</kbd> type · <kbd>A</kbd> panel · <kbd>M</kbd> mute · <kbd>Esc</kbd> stop · <kbd>?</kbd> help.

## Experiments and videos

- `train/experiments/` reproduces every number in the labs (ablations, scaling, learning-rate sweep, overfitting, gradients,
  depth); results land in `public/research/*.json`. See its README for recipes and schemas.
- `video/` builds the explainer series with Manim; each scene declares its own metadata and `python3 video/build.py <id>`
  renders, narrates, encodes and indexes it. See `video/README.md`.

## The model

`train/` contains everything: a synthetic "tiny world" corpus (`corpus.py`), a BPE tokenizer (`bpe.py`), and a GPT
(`train.py`: 3 layers, 4 heads, d=64, 256 tokens). Training takes about 80 s on a CPU and exports weights to
`public/model/`. `src/model/gpt.ts` re-implements the forward pass in TypeScript and records every intermediate
value. It matches PyTorch to 4 decimal places.

```bash
pip install torch numpy && npm run train   # retrain (optional)
```

## Develop & deploy

```bash
npm install
echo "DEEPSEEK_API_KEY=sk-..." > .dev.vars    # or ANTHROPIC_API_KEY=... (Claude is used if set)
npm run build && npx wrangler pages dev dist  # http://localhost:8788
npm run deploy                                # → Cloudflare Pages project "topdownai"
```

Production secret (one time; the tutor stays off until this is set):

```bash
npx wrangler pages secret put DEEPSEEK_API_KEY --project-name topdownai
```

Learner data lives in your browser's localStorage. Use Export/Import on *My path* to move it between devices.
