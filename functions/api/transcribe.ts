import type { Env } from "../_lib/llm";

// Speech → text with Whisper on Workers AI. Body: raw audio (webm/ogg/mp4/mp3/wav).
const VOCAB = "Transformer, token, tokenizer, embedding, attention head, query, key, value, softmax, logit, logit lens, " +
  "residual stream, MLP, neuron, layer norm, temperature, top-k, sampling, gradient descent, backprop, loss, TinyGPT, BPE.";

function toBase64(buf: ArrayBuffer) {
  const bytes = new Uint8Array(buf); let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export const onRequestPost: PagesFunction<Env> = async ({ request, env }) => {
  if (!env.AI) return new Response("Workers AI binding missing", { status: 500 });
  const buf = await request.arrayBuffer();
  if (buf.byteLength < 200) return Response.json({ text: "" });
  if (buf.byteLength > 12_000_000) return new Response("Recording too long", { status: 413 });
  try {
    const out = (await env.AI.run("@cf/openai/whisper-large-v3-turbo" as keyof AiModels, {
      audio: toBase64(buf), language: "en", initial_prompt: VOCAB, vad_filter: true,
    } as never)) as { text?: string };
    return Response.json({ text: (out.text ?? "").trim() });
  } catch (e) {
    return new Response(`Transcription failed: ${(e as Error).message}`, { status: 500 });
  }
};
