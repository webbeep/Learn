import { useEffect, useRef, useState } from "react";
import { ask, learner, useStore, type AskItem } from "../lib/store";
import { parseAnswer, recordQA, streamAsk, type Turn } from "../lib/api";
import { Markdown } from "./Markdown";
import { suggestionsFor } from "./suggest";

export function AskPanel() {
  const open = useStore(ask, (s) => s.open);
  const selection = useStore(ask, (s) => s.selection);
  const region = useStore(ask, (s) => s.region);
  const regionMode = useStore(ask, (s) => s.regionMode);
  const scene = useStore(ask, (s) => `${s.page}${s.scene ? " › " + s.scene : ""}`);
  const draft = useStore(ask, (s) => s.draft);
  const [thread, setThread] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => { if (draft) { setInput(draft); ask.set({ draft: "" }); } }, [draft]);
  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight }); }, [thread]);

  async function send(q: string) {
    q = q.trim(); if (!q || busy) return;
    const sel: AskItem[] = ask.get().selection;
    const prior = thread;
    setThread([...prior, { role: "user", content: q, selection: sel }, { role: "assistant", content: "" }]);
    setInput(""); setBusy(true);
    abort.current = new AbortController();
    try {
      const full = await streamAsk(q, prior, (t) => setThread((th) => [...th.slice(0, -1), { role: "assistant", content: t }]), abort.current.signal);
      const { followups } = parseAnswer(full);
      setThread((th) => [...th.slice(0, -1), { role: "assistant", content: full, followups }]);
      recordQA(q, full, sel);
    } catch (e) {
      const msg = (e as Error).name === "AbortError" ? "(stopped)" : `⚠️ ${(e as Error).message}`;
      setThread((th) => [...th.slice(0, -1), { role: "assistant", content: msg, error: true }]);
    } finally { setBusy(false); }
  }

  const last = thread[thread.length - 1];
  const chips = last?.followups?.length ? last.followups : suggestionsFor(selection.map((s) => s.id));
  if (!open) return null;
  return (
    <aside className="ask-panel">
      <header>
        <div className="ask-title"><b>Ask the tutor</b><span className="muted"> · {scene || "anywhere"}</span></div>
        <div className="row">
          {thread.length > 0 && <button className="ghost" onClick={() => setThread([])} title="Start a new conversation">New</button>}
          <button className="ghost" onClick={() => ask.set({ open: false })} aria-label="Close">✕</button>
        </div>
      </header>

      <div className="ask-context">
        {selection.length ? (
          <div className="sel-list">
            {selection.slice(0, 6).map((s) => (
              <span key={s.id} className="sel-chip" onMouseEnter={() => ask.set({ highlight: s.id })} onMouseLeave={() => ask.set({ highlight: null })}>
                {s.label}
                <button onClick={() => ask.set({ selection: selection.filter((x) => x.id !== s.id) })}>×</button>
              </span>
            ))}
            {selection.length > 6 && <span className="muted">+{selection.length - 6} more</span>}
          </div>
        ) : <div className="muted small">Click anything in the visual (shift-click for several), or highlight a region. The tutor sees exactly what you selected plus everything on screen.</div>}
        {region && <div className="muted small">▭ {region}</div>}
        <div className="row">
          <button className={regionMode ? "primary" : ""} onClick={() => ask.set({ regionMode: !regionMode })}>✏️ {regionMode ? "Drag over the visual…" : "Highlight region"}</button>
          {selection.length > 0 && <button className="ghost" onClick={() => ask.set({ selection: [], region: "" })}>Clear</button>}
        </div>
      </div>

      <div className="thread" ref={scroller}>
        {thread.length === 0 && <Intro />}
        {thread.map((t, i) => (
          <div key={i} className={`msg ${t.role}${t.error ? " error" : ""}`}>
            {t.role === "user" ? (<>
              {t.selection?.length ? <div className="muted small">about: {t.selection.slice(0, 3).map((s) => s.label).join(" · ")}{t.selection.length > 3 ? " …" : ""}</div> : null}
              <div>{t.content}</div>
            </>) : t.content ? <Markdown text={parseAnswer(t.content).body} /> : <div className="typing"><i /><i /><i /></div>}
          </div>
        ))}
      </div>

      <div className="chips">
        {chips.map((c) => <button key={c} className="chip" disabled={busy} onClick={() => send(c)}>{c}</button>)}
      </div>
      <form className="ask-input" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <textarea value={input} rows={2} placeholder="Ask anything… (Enter to send)" onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }} />
        {busy ? <button type="button" onClick={() => abort.current?.abort()}>Stop</button> : <button className="primary" type="submit">Ask</button>}
      </form>
    </aside>
  );
}

function Intro() {
  const p = useStore(learner, (s) => s.profile);
  return (
    <div className="intro muted small">
      <p>I can see what's on your screen and the exact numbers inside the model. I'll point at things with <span className="ptr-chip">◎ chips</span> — hover to highlight them — and suggest <span className="try-chip">▶ experiments</span> you can run.</p>
      {p?.summary && <p>What I know about you so far: <i>{p.summary}</i></p>}
    </div>
  );
}

/** Small "ask about this" popover shown on click when the panel is closed. */
export function AskPopover() {
  const sel = useStore(ask, (s) => s.selection);
  const open = useStore(ask, (s) => s.open);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  useEffect(() => {
    const h = (e: Event) => setPos((e as CustomEvent).detail);
    window.addEventListener("ask-popover", h); return () => window.removeEventListener("ask-popover", h);
  }, []);
  useEffect(() => { if (open || !sel.length) setPos(null); }, [open, sel.length]);
  if (!pos || open || !sel.length) return null;
  const x = Math.min(pos.x, window.innerWidth - 300), y = Math.min(pos.y + 12, window.innerHeight - 190);
  return (
    <div className="popover" style={{ left: Math.max(8, x), top: y }} onClick={(e) => e.stopPropagation()}>
      <div className="small"><b>{sel[0].label}</b>{sel.length > 1 ? <span className="muted"> +{sel.length - 1}</span> : null}</div>
      {suggestionsFor(sel.map((s) => s.id)).slice(0, 2).map((q) => (
        <button key={q} className="chip" onClick={() => { ask.set({ open: true, draft: q }); }}>{q}</button>
      ))}
      <div className="row">
        <button className="primary" onClick={() => ask.set({ open: true })}>💬 Ask about this</button>
        <button className="ghost" onClick={() => { setPos(null); ask.set({ selection: [] }); }}>✕</button>
      </div>
    </div>
  );
}
