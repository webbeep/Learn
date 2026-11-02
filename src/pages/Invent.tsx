import { useCallback, useEffect, useState } from "react";
import { setScreen } from "../lib/store";
import { ResearchLoop } from "../labs/InventLoop";
import { CardBuilder, IdeaBoard } from "../labs/InventCards";
import type { Idea } from "../labs/ideas";

export function Invent() {
  const [seed, setSeed] = useState<{ idea: Idea; n: number } | null>(null);
  const [parts, setParts] = useState({ loop: "", cards: "" });
  const onLoop = useCallback((s: string) => setParts((p) => (p.loop === s ? p : { ...p, loop: s })), []);
  const onCards = useCallback((s: string) => setParts((p) => (p.cards === s ? p : { ...p, cards: s })), []);
  useEffect(() => {
    setScreen("Invent your own", seed ? `Idea: ${seed.idea.title}` : "Research workspace", [
      "A guided research workspace: the research loop (hypothesis → smallest experiment → baseline → change one thing → compare across seeds → scale up), an idea board of frontier directions (mixture of experts, state-space models, linear attention, RoPE/long context, retrieval, diffusion LMs, test-time compute, distillation), and experiment cards that export a runnable PyTorch train.py.",
      parts.loop, seed ? `Learner just picked the idea "${seed.idea.title}".` : "", parts.cards,
    ].filter(Boolean).join("\n\n"));
  }, [seed, parts]);

  return (
    <div className="space-y-8 pb-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Invent your own</h1>
      </div>
      <ResearchLoop onSummary={onLoop} />
      <IdeaBoard onUse={(idea) => setSeed((s) => ({ idea, n: (s?.n ?? 0) + 1 }))} />
      <CardBuilder seed={seed} onSummary={onCards} />
    </div>
  );
}
