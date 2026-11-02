import { useState } from "react";
import { phase, show, tokColor } from "../lib/viz";
import type { SceneProps } from "./types";

export function TokensScene({ m, step, si, t }: SceneProps) {
  const [all, setAll] = useState(false);
  const pieces = m.tok.encode(m.tok.decode(step.ctx));
  // trust ids from ctx (identical to re-encoding in practice, but ctx is the truth)
  const toks = step.ctx.map((id, i) => ({ id, text: m.meta.vocab[id], merges: pieces[i]?.id === id ? pieces[i].merges : [] }));
  const shown = Math.ceil(phase(t, 0.05, 0.6) * toks.length);
  const used = new Map<number, number>(); toks.forEach((x) => used.set(x.id, (used.get(x.id) ?? 0) + 1));
  return (
    <div className="scene tokens-scene">
      <div className="lane">
        <div className="lane-label">Text</div>
        <div className="raw-text mono">
          {toks.map((x, i) => (
            <span key={i} className={i === toks.length - 1 && si > 0 ? "fresh" : ""} style={{ background: i < shown ? tokColor(x.id, 22) : "transparent" }}>{x.text}</span>
          ))}
        </div>
      </div>
      <div className="arrow-down">↓ {toks.length} tokens</div>
      <div className="lane">
        <div className="lane-label">Tokens</div>
        <div className="token-row">
          {toks.map((x, i) => (
            <div key={i} className={`tok ${i < shown ? "in" : ""} ${i === toks.length - 1 && si > 0 ? "fresh" : ""}`} style={{ transitionDelay: `${(i % 30) * 15}ms`, borderColor: tokColor(x.id) }}
              data-ask-id={`tok-${i}`}
              data-ask={`token #${i} "${show(x.text)}" = id ${x.id}${x.merges.length ? `, built by merges ${x.merges.join(", ")}` : ", a single character"}`}>
              <span className="mono" style={{ color: tokColor(x.id, 75) }}>{show(x.text)}</span>
              <small>{x.id}</small>
            </div>
          ))}
        </div>
      </div>
      <button className="lane-label underline-offset-2 hover:underline" onClick={() => setAll(!all)}>{all ? "Hide" : "Show"} all {m.cfg.vocab_size} tokens it knows</button>
      {all && <div className="lane">
        <div className="vocab-grid">
          {m.meta.vocab.map((v, id) => (
            <div key={id} className={`vcell ${used.has(id) ? "on" : ""}`} style={used.has(id) ? { background: tokColor(id, 45) } : undefined}
              data-ask-id={`vocab-${id}`} data-ask={`vocabulary entry id ${id} = "${show(v)}"${used.has(id) ? ` (appears ${used.get(id)}× in the text)` : " (not in the text)"}`}
              title={`${id}: ${show(v)}`}>{show(v)}</div>
          ))}
        </div>
      </div>}
    </div>
  );
}
