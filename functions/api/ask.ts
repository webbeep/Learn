import { streamChat, type Env, type Msg } from "../_lib/llm";

interface Body {
  question: string;
  thread?: Msg[];
  context?: {
    page?: string; scene?: string; summary?: string;
    selection?: { id: string; label: string }[];
    elements?: { id: string; label: string }[];
    region?: string;
  };
  learner?: { goal?: string; depth?: string; length?: string; profile?: string; recent?: string[] };
}

const SYSTEM = `You are the tutor inside "Top-Down AI", a visual, hands-on app for someone who learns TOP-DOWN:
they start from seeing a working system, poke at it, and only dive into fundamentals when they hit a question.
Their long-term goal: train models from scratch and eventually invent a new kind of AI model / training framework.

THE MODEL IN THE APP is a real, tiny GPT running in the learner's browser ("TinyGPT"):
- 3 transformer blocks, 4 attention heads each, d_model=64, MLP width 256, context 64 tokens, ~170k parameters.
- Byte-pair-encoding tokenizer with 256 tokens (lowercase letters, a little punctuation; spaces are attached to the front of words, shown as ␣).
- Pre-LayerNorm GPT-2 style blocks, GELU MLP, learned positional embeddings, weight-tied unembedding.
- Trained for 3000 steps (~80 seconds on a CPU) on a synthetic "tiny world" corpus: kids-story sentences like
  "mia has a red ball. tom has a blue kite. mia gives the ball to tom. now tom has the red ball.",
  pet sounds, days of the week, counting, and "q: ... a: ..." questions. Final loss ≈ 0.46 nats/token.
Every number the learner sees is real and comes from this model. Real LLMs work the same way, just scaled up
(e.g. ~100k-token vocabularies, thousands of dimensions, 100+ layers, trillions of training tokens).

HOW TO ANSWER
- Ground your answer in the CONTEXT: the exact tokens, weights, probabilities and the learner's SELECTION. Quote numbers.
- Point at things on screen with [[element-id]] using ONLY ids listed in the context. The app turns these into
  clickable chips that highlight the element. Use 1–4 of them when helpful.
- Suggest hands-on experiments as [[try: some prompt text]] (lowercase, the tiny model only knows its tiny world).
  The app turns these into buttons that run the prompt.
- Intuition first, then (if useful) the mechanism, then how it scales to real LLMs. Use analogies, small markdown
  tables, short bullet lists. No walls of text. Use LaTeX-free plain math like softmax(x)_i = e^x_i / Σ e^x_j.
- Be honest about what a tiny model can't do and where interpretations are uncertain.
- If they ask about something deeper (backprop, gradient descent, scaling laws, GPUs, new architectures), give the
  top-down intuition anchored in what they're looking at, and say which level of the app will go deeper.
- If the learner wants to SEE something the app can't show yet, add a line: BUILD: <one-line feature idea>.
- Always end with exactly one line: FOLLOWUPS: <question 1> | <question 2> | <question 3>
  (short, curious next questions tailored to this learner).`;

export const onRequestPost: PagesFunction<Env> = async ({ request, env }) => {
  try {
    const b = (await request.json()) as Body;
    const c = b.context ?? {}, l = b.learner ?? {};
    const lines = [
      `LEARNER: goal=${l.goal ?? "train models from scratch, invent new AI"}; prefers depth=${l.depth ?? "balanced"}, length=${l.length ?? "medium"}.`,
      l.profile ? `LEARNER PROFILE (from past questions): ${l.profile}` : "",
      l.recent?.length ? `RECENT QUESTIONS: ${l.recent.join(" / ")}` : "",
      `CONTEXT — page: ${c.page ?? "?"}; scene: ${c.scene ?? "?"}`,
      c.summary ? `WHAT IS ON SCREEN:\n${c.summary}` : "",
      c.region ? `THE LEARNER HIGHLIGHTED A REGION: ${c.region}` : "",
      c.selection?.length ? `SELECTION (what the learner clicked/highlighted):\n${c.selection.map((s) => `- [${s.id}] ${s.label}`).join("\n")}` : "SELECTION: nothing specific",
      c.elements?.length ? `OTHER ELEMENTS ON SCREEN (ids you may cite):\n${c.elements.map((s) => `- [${s.id}] ${s.label}`).join("\n")}` : "",
    ].filter(Boolean).join("\n");
    const messages: Msg[] = [...(b.thread ?? []).slice(-8), { role: "user", content: `${lines}\n\nQUESTION: ${b.question}` }];
    const stream = await streamChat(env, SYSTEM, messages, l.length === "short" ? 700 : 1600);
    return new Response(stream, { headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" } });
  } catch (e) {
    return new Response(String((e as Error).message ?? e), { status: 500 });
  }
};
