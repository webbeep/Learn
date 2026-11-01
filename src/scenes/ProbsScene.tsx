import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { useState } from "react";
import { softmax, topK } from "../model/gpt";
import { pct, phase, show, tokColor } from "../lib/viz";
import type { SceneProps } from "./types";

export function ProbsScene({ m, step, si, t, settings, regenerate }: SceneProps) {
  const [temp, setTemp] = useState(settings.temperature || 1);
  const logits = step.trace.logits;
  const probs = softmax(logits, Math.max(temp, 0.05));
  const top = topK(probs, 14);
  const rest = 1 - top.reduce((a, b) => a + b.p, 0);
  const maxL = Math.max(...top.map((p) => logits[p.id])), minL = Math.min(...Array.from(logits));
  const grow = phase(t, 0.1, 0.6), soft = phase(t, 0.4, 0.9);
  return (
    <div className="scene probs-scene">
      <div className="lane-label">The final vector is compared with every one of the {m.cfg.vocab_size} token vectors → one score ("logit") per token → softmax turns scores into probabilities.</div>
      <div className="prob-table">
        <div className="ph"><span>token</span><span>score (logit)</span><span /><span>probability @ temperature {temp.toFixed(2)}</span></div>
        {top.map((p) => {
          const lg = logits[p.id];
          return (
            <div className="pr" key={p.id} data-ask-id={`prob-${p.id}`} data-ask={`candidate "${show(m.meta.vocab[p.id])}": logit ${lg.toFixed(2)}, probability ${pct(p.p)} at temperature ${temp.toFixed(2)}`}>
              <span className="mono" style={{ color: tokColor(p.id, 75) }}>{show(m.meta.vocab[p.id])}</span>
              <span className="bar"><i style={{ width: `${grow * Math.max(1, ((lg - minL) / (maxL - minL)) * 100)}%`, background: "var(--muted-foreground)" }} /><b>{lg.toFixed(1)}</b></span>
              <span className="arrow">→</span>
              <span className="bar"><i style={{ width: `${soft * Math.max(0.5, p.p * 100)}%`, background: tokColor(p.id, 50) }} /><b>{pct(p.p)}</b></span>
            </div>
          );
        })}
        <div className="pr muted small" data-ask-id="prob-rest" data-ask={`the other ${m.cfg.vocab_size - top.length} tokens together get ${pct(rest)}`}><span>…{m.cfg.vocab_size - top.length} others</span><span /><span /><span>{pct(rest)} total</span></div>
      </div>
      <div className="temp-box">
        <label>🌡️ What-if temperature <b>{temp.toFixed(2)}</b>
          <Slider min={0.05} max={3} step={0.05} value={[temp]} onValueChange={([v]) => setTemp(v)} className="my-2" />
        </label>
        <div className="small muted">Low = confident & repetitive. High = flatter, more surprising (and more wrong). softmax(score / temperature).</div>
        {Math.abs(temp - settings.temperature) > 0.01 && <Button size="sm" onClick={() => regenerate({ temperature: temp }, si)}><RefreshCw />Regenerate at {temp.toFixed(2)}</Button>}
      </div>
    </div>
  );
}
