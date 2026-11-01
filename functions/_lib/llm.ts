// Provider-agnostic chat call. Uses Claude if ANTHROPIC_API_KEY is set,
// otherwise DeepSeek (OpenAI-compatible API).
export interface Env { AI?: Ai; DEEPSEEK_API_KEY?: string; ANTHROPIC_API_KEY?: string; ANTHROPIC_MODEL?: string; DEEPSEEK_MODEL?: string }
export interface Msg { role: "user" | "assistant"; content: string }

export function provider(env: Env) {
  if (env.ANTHROPIC_API_KEY) return "anthropic";
  if (env.DEEPSEEK_API_KEY) return "deepseek";
  return null;
}

/** Returns a ReadableStream of plain UTF-8 text deltas. */
export async function streamChat(env: Env, system: string, messages: Msg[], maxTokens = 1400): Promise<ReadableStream<Uint8Array>> {
  const p = provider(env);
  if (!p) throw new Error("No LLM key configured. Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY as a Pages secret.");
  const res = p === "anthropic"
    ? await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "x-api-key": env.ANTHROPIC_API_KEY!, "anthropic-version": "2023-06-01", "content-type": "application/json" },
        body: JSON.stringify({ model: env.ANTHROPIC_MODEL || "claude-sonnet-5-5", max_tokens: maxTokens, system, messages, stream: true }),
      })
    : await fetch("https://api.deepseek.com/chat/completions", {
        method: "POST",
        headers: { authorization: `Bearer ${env.DEEPSEEK_API_KEY}`, "content-type": "application/json" },
        body: JSON.stringify({ model: env.DEEPSEEK_MODEL || "deepseek-chat", max_tokens: maxTokens, stream: true,
          messages: [{ role: "system", content: system }, ...messages] }),
      });
  if (!res.ok || !res.body) throw new Error(`${p} error ${res.status}: ${(await res.text()).slice(0, 300)}`);

  const enc = new TextEncoder(), dec = new TextDecoder();
  let buf = "";
  return res.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, ctl) {
      buf += dec.decode(chunk, { stream: true });
      const lines = buf.split("\n"); buf = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        try {
          const j = JSON.parse(data);
          const t = p === "anthropic" ? (j.type === "content_block_delta" ? j.delta?.text : undefined) : j.choices?.[0]?.delta?.content;
          if (t) ctl.enqueue(enc.encode(t));
        } catch { /* ignore keep-alives */ }
      }
    },
  }));
}

export async function completeJSON(env: Env, system: string, user: string): Promise<unknown> {
  const s = await streamChat(env, system + "\nRespond with ONLY a JSON object, no prose, no code fences.", [{ role: "user", content: user }], 1500);
  const text = await new Response(s).text();
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) throw new Error("Model did not return JSON");
  return JSON.parse(m[0]);
}
