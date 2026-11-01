import { useEffect, useRef, useState } from "react";
import { ask, learner, useStore } from "../lib/store";
import { compressSelection, parseAnswer, viewportNote, visibleElements } from "../lib/api";
import { convo, currentPointer, listenAndAsk, newConversation, sendQuestion, stopAll } from "../lib/conversation";
import { speakText, stopListening, stopSpeaking, voice } from "../lib/voice";
import { Markdown } from "./Markdown";
import { suggestionsFor } from "./suggest";
import { VoiceSettings } from "./VoiceSettings";

export function AskPanel() {
  const open = useStore(ask, (s) => s.open);
  const selection = useStore(ask, (s) => s.selection);
  const region = useStore(ask, (s) => s.region);
  const regionMode = useStore(ask, (s) => s.regionMode);
  const scene = useStore(ask, (s) => `${s.page}${s.scene ? " › " + s.scene : ""}`);
  const draft = useStore(ask, (s) => s.draft);
  const thread = useStore(convo, (s) => s.thread);
  const busy = useStore(convo, (s) => s.busy);
  const listening = useStore(voice, (s) => s.listening || s.transcribing);
  const speaking = useStore(voice, (s) => s.speaking);
  const interim = useStore(voice, (s) => s.interim);
  const [input, setInput] = useState("");
  const [tab, setTab] = useState<"chat" | "settings" | "sees">("chat");
  const scroller = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { if (draft) { setInput(draft); ask.set({ draft: "" }); setTimeout(() => inputRef.current?.focus(), 0); } }, [draft]);
  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight }); }, [thread, interim]);
  useEffect(() => {
    const f = () => { setTab("chat"); setTimeout(() => inputRef.current?.focus(), 30); };
    window.addEventListener("focus-ask", f); return () => window.removeEventListener("focus-ask", f);
  }, []);

  const send = (q: string) => { if (!q.trim()) return; setInput(""); sendQuestion(q, "text"); };
  const last = thread[thread.length - 1];
  const chips = last?.followups?.length ? last.followups : suggestionsFor(selection.map((s) => s.id));
  if (!open) return null;
  return (
    <aside className="ask-panel">
      <header>
        <div className="ask-title"><b>Ask the tutor</b><span className="muted"> · {scene || "anywhere"}</span></div>
        <div className="row">
          <button className={`ghost ${tab === "sees" ? "on" : ""}`} onClick={() => setTab(tab === "sees" ? "chat" : "sees")} title="What the tutor can see right now">👁</button>
          <button className={`ghost ${tab === "settings" ? "on" : ""}`} onClick={() => setTab(tab === "settings" ? "chat" : "settings")} title="Voice & highlight settings">⚙</button>
          {thread.length > 0 && <button className="ghost" onClick={newConversation} title="Start a new conversation">New</button>}
          <button className="ghost" onClick={() => ask.set({ open: false })} aria-label="Close">✕</button>
        </div>
      </header>

      {tab === "settings" ? <div className="thread"><VoiceSettings /></div> : tab === "sees" ? <div className="thread"><WhatItSees /></div> : (<>
        <div className="ask-context">
          {selection.length ? (
            <div className="sel-list">
              {selection.slice(0, 6).map((s) => (
                <span key={s.id} className="sel-chip" onMouseEnter={() => ask.set({ highlight: s.id })} onMouseLeave={() => ask.set({ highlight: null })}>
                  {s.label.length > 140 ? s.label.slice(0, 138) + "…" : s.label}
                  <button onClick={() => ask.set({ selection: selection.filter((x) => x.id !== s.id) })}>×</button>
                </span>
              ))}
              {selection.length > 6 && <span className="muted">+{selection.length - 6} more</span>}
            </div>
          ) : <div className="muted small">Click anything, highlight text, or drag a box (<kbd>H</kbd>). With nothing selected, the tutor uses whatever your mouse is over and everything visible in the window.</div>}
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
                {t.selection?.length ? <div className="muted small">about: {t.selection.slice(0, 3).map((s) => s.label.slice(0, 90)).join(" · ")}{t.selection.length > 3 ? " …" : ""}</div> : null}
                <div>{t.via === "voice" && "🎙 "}{t.content}</div>
              </>) : t.content ? (<>
                <Markdown text={parseAnswer(t.content).body} />
                {!t.error && !(busy && i === thread.length - 1) && <button className="ghost small replay" onClick={() => speakText(parseAnswer(t.content).body)}>🔊 read aloud</button>}
              </>) : <div className="typing"><i /><i /><i /></div>}
            </div>
          ))}
          {listening && <div className="msg user listening-bubble">🎙 {interim || "Listening…"}</div>}
        </div>

        <div className="chips">
          {chips.map((c) => <button key={c} className="chip" onClick={() => sendQuestion(c, "text")}>{c}</button>)}
        </div>
        <form className="ask-input" onSubmit={(e) => { e.preventDefault(); send(input); }}>
          <button type="button" className={`mic ${listening ? "on" : ""}`} title="Talk (V) — click, or hold V to push-to-talk"
            onClick={() => (listening ? stopListening() : listenAndAsk())}>{listening ? "■" : "🎙"}</button>
          <textarea ref={inputRef} value={input} rows={2} placeholder="Type, or press V and talk…" onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } if (e.key === "Escape") (e.target as HTMLElement).blur(); }} />
          {busy || speaking ? <button type="button" onClick={() => { stopAll(); stopSpeaking(); }}>Stop</button> : <button className="primary" type="submit">Ask</button>}
        </form>
      </>)}
    </aside>
  );
}

function Intro() {
  const p = useStore(learner, (s) => s.profile);
  return (
    <div className="intro muted small">
      <p>I can see what's in your window, what you've selected or highlighted, and what your mouse is over, along with the exact numbers inside the model. Ask by typing, or press <kbd>V</kbd> and just talk. I'll answer out loud if you like (⚙).</p>
      <p>I point at things with <span className="ptr-chip">◎</span> chips (hover them to light things up) and suggest <span className="try-chip">▶ experiments</span>. Press <kbd>?</kbd> for shortcuts.</p>
      {p?.summary && <p>What I know about you so far: <i>{p.summary}</i></p>}
    </div>
  );
}

/** Transparency: show exactly what context the tutor receives. */
function WhatItSees() {
  const a = useStore(ask, (s) => s);
  const [, tick] = useState(0);
  useEffect(() => { const iv = setInterval(() => tick((x) => x + 1), 1000); return () => clearInterval(iv); }, []);
  const els = visibleElements(), ptr = currentPointer();
  return (
    <div className="sees small">
      <p className="muted">This is sent with every question. It updates live as you scroll, click and move the mouse.</p>
      <h4>Page</h4><p>{a.page} {a.scene && `› ${a.scene}`}</p>
      <h4>Window</h4><p>{viewportNote()}</p>
      <h4>On screen</h4><pre>{a.summary || "—"}</pre>
      <h4>Your selection</h4><p>{a.selection.length ? compressSelection(a.selection).map((s) => s.label).join(" · ") : "nothing"}</p>
      <h4>Mouse is over</h4><p>{ptr ? `${ptr.label} (${ptr.ago}s ago)` : "nothing recent"}</p>
      <h4>{els.length} visible elements it can point at</h4>
      <ul>{els.slice(0, 40).map((e) => <li key={e.id}><code>{e.id}</code> {e.label.slice(0, 100)}</li>)}</ul>
    </div>
  );
}

/** Small "ask about this" popover shown after a click / highlight when the panel is closed. */
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
  const x = Math.min(pos.x, window.innerWidth - 310), y = Math.min(pos.y + 12, window.innerHeight - 200);
  const go = (fn: () => void) => { setPos(null); ask.set({ open: true }); fn(); };
  return (
    <div className="popover" style={{ left: Math.max(8, x), top: y }} onMouseUp={(e) => e.stopPropagation()} onClick={(e) => e.stopPropagation()}>
      <div className="small"><b>{sel[0].label.slice(0, 120)}</b>{sel.length > 1 ? <span className="muted"> +{sel.length - 1}</span> : null}</div>
      {suggestionsFor(sel.map((s) => s.id)).slice(0, 2).map((q) => (
        <button key={q} className="chip" onClick={() => go(() => sendQuestion(q, "text"))}>{q}</button>
      ))}
      <div className="row">
        <button className="primary" onClick={() => go(() => listenAndAsk())} title="V">🎙 Talk</button>
        <button onClick={() => go(() => window.dispatchEvent(new Event("focus-ask")))} title="/">⌨️ Type</button>
        <button onClick={() => go(() => sendQuestion("Explain this.", "text"))} title="E">✨ Explain</button>
        <button className="ghost" onClick={() => { setPos(null); ask.set({ selection: [] }); }}>✕</button>
      </div>
    </div>
  );
}
