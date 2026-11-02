import { completeJSON, type Env } from "../_lib/llm";

// Rebuilds the learner profile from their question history. The app uses it to
// tailor answers and to recommend (and request!) what to build next.
const SYSTEM = `You maintain the learner model for "Top-Down AI", a visual app teaching how LLMs work, top-down,
via a real tiny GPT running in the browser. Learner's long-term goal: train models from scratch and invent a new AI
model/training framework. App sections: curriculum (18 chapters: tokens, embeddings, attention, mlp, residual-stream, next-token, loss, backprop,
optimizers, train-your-own, depth, arch-lab, scaling, chatgpt, research-loop, frontier), labs: "xray" (inference),
"training" (real training run), "train" (train in the browser), "backprop", "lab" (ablations), "scaling", "invent",
and "watch" (videos).

Given the previous profile and recent Q&A, return JSON:
{
 "summary": "2-3 sentences: who this learner is, how they think, where they are",
 "knows": ["concepts they seem to grasp"],
 "confusions": ["open confusions / misconceptions to address"],
 "interests": ["what they keep gravitating to"],
 "style": "how to explain to them (one sentence)",
 "next": [{"title": "...", "why": "...", "where": "xray|training|train|backprop|lab|scaling|invent|watch|learn/<chapter-id>", "try": "optional prompt for the tiny model"}],
 "build": ["new visualization/feature ideas for the app, derived from their questions"]
}
"try" prompts run on the tiny model, so they MUST be lowercase tiny-world text built from words like: mia tom lily sam ben zoe max ava leo ruby finn nora,
cat dog fox bird frog bear owl fish duck rabbit, ball kite hat book cup box drum shoe boat key, red blue green yellow pink black white orange purple,
park forest river school garden beach house hill, has gives now the a q: what color is ... a:, once upon a time, monday..sunday, one..ten.
Keep lists ≤ 6 items, concise.`;

export const onRequestPost: PagesFunction<Env> = async ({ request, env }) => {
  try {
    const b = (await request.json()) as { previous?: unknown; history?: { q: string; a: string; where: string }[] };
    const hist = (b.history ?? []).slice(-15).map((h) => `[${h.where}] Q: ${h.q}\nA (excerpt): ${h.a.slice(0, 400)}`).join("\n\n");
    const json = await completeJSON(env, SYSTEM, `PREVIOUS PROFILE: ${JSON.stringify(b.previous ?? null)}\n\nRECENT Q&A:\n${hist}`);
    return Response.json(json);
  } catch (e) {
    return new Response(String((e as Error).message ?? e), { status: 500 });
  }
};
