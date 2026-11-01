import type { Env } from "../_lib/llm";

// Text → natural speech (mp3) with Deepgram Aura-2 on Workers AI.
const SPEAKERS = new Set(["luna", "asteria", "thalia", "helena", "orion", "arcas", "apollo", "zeus"]);

export const onRequestPost: PagesFunction<Env> = async ({ request, env }) => {
  if (!env.AI) return new Response("Workers AI binding missing", { status: 500 });
  const { text, voice } = (await request.json()) as { text?: string; voice?: string };
  const t = (text ?? "").slice(0, 1500).trim();
  if (!t) return new Response("empty", { status: 400 });
  try {
    const out: unknown = await env.AI.run("@cf/deepgram/aura-2-en" as keyof AiModels,
      { text: t, speaker: SPEAKERS.has(voice ?? "") ? voice : "luna", encoding: "mp3" } as never, { returnRawResponse: true } as never);
    const headers = { "content-type": "audio/mpeg", "cache-control": "no-store" };
    if (out instanceof Response) return new Response(out.body, { headers });
    if (out instanceof ReadableStream) return new Response(out, { headers });
    const b64 = (out as { audio?: string }).audio;
    if (b64) return new Response(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)), { headers });
    return new Response("Unexpected TTS output", { status: 500 });
  } catch (e) {
    return new Response(`Speech failed: ${(e as Error).message}`, { status: 500 });
  }
};
