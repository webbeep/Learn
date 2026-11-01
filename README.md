# 🔭 Top-Down AI

Learn how AI models work **top-down**: start with a real model running in front of you, poke it, and dive into
the fundamentals only when you hit a question. Every visual is clickable, and an AI tutor answers questions about
exactly what you selected.

**Live:** https://learn-ai-7hs.pages.dev

## What's inside

| Section | What you do |
|---|---|
| 🔬 **X-ray** | Type a prompt and press ▶. A real 170k-parameter GPT runs **in your browser** and you watch it, like a video, go through text → tokens → vectors → attention → layers (logit lens + MLP neurons) → probabilities → dice roll, then loop. Pause anywhere, click any number, or ✏️ drag a box over a region to ask about it. |
| 📉 **Training** | The real loss curve of that same model, plus what it wrote at steps 0 / 25 / 100 / 300 / 1000 / 3000. |
| 🎬 **Watch** | Hand-picked 3Blue1Brown and Karpathy videos. Pause, and the tutor knows which video you're on and the timestamp. |
| 🧭 **My path** | The tutor's model of you: what you understand, what's still fuzzy, and what to try next, built from your questions. Also a **build list** of features you wish existed, which you can copy as a request for the next Claude Code session. That's how the app evolves. |

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
npm run deploy                                # → Cloudflare Pages project "learn-ai"
```

Production secret (one time; the tutor stays off until this is set):

```bash
npx wrangler pages secret put DEEPSEEK_API_KEY --project-name learn-ai
```

Learner data lives in your browser's localStorage. Use Export/Import on *My path* to move it between devices.
