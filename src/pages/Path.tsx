import { useEffect, useState } from "react";
import { addWish, ask, learner, setScreen, useStore } from "../lib/store";
import { refreshProfile } from "../lib/api";
import { Markdown } from "../ask/Markdown";

export function Path() {
  const l = useStore(learner, (s) => s);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [wish, setWish] = useState("");
  useEffect(() => setScreen("My path", "", `Learner has asked ${l.history.length} questions. Profile: ${l.profile?.summary ?? "none yet"}. Build list: ${l.wishes.map((w) => w.text).join("; ")}`), [l]);
  const p = l.profile;
  const exportPrompt = [
    "Here is my learner profile and feature wishlist from my Top-Down AI app. Please evolve the app (webbeep/Learn) to build the top items, keeping the same style:",
    "", `Goal: ${l.goal}`, p ? `Profile: ${p.summary}\nStyle: ${p.style}\nConfusions: ${p.confusions.join("; ")}\nInterests: ${p.interests.join("; ")}` : "",
    "", "Wishlist:", ...l.wishes.filter((w) => !w.done).map((w) => `- ${w.text}  (from: ${w.source})`),
    "", "Recent questions:", ...l.history.slice(-12).map((h) => `- [${h.where}] ${h.q}`),
  ].join("\n");

  return (
    <div className="page path">
      <h1>My path</h1>
      <p className="lead">The app keeps a picture of how you think, built from your questions. It uses it to tailor every answer — and it turns what you wish you could see into a build list for the next version of this app.</p>

      <div className="two-col">
        <div className="card">
          <div className="row between"><div className="lane-label">How the tutor sees you</div>
            <button disabled={busy || !l.history.length} onClick={async () => { setBusy(true); setErr(""); try { await refreshProfile(); } catch (e) { setErr(String((e as Error).message)); } setBusy(false); }}>{busy ? "Thinking…" : "↻ Update from my questions"}</button></div>
          {err && <div className="error small">{err}</div>}
          {p ? (<>
            <p>{p.summary}</p>
            <p className="small"><b>Explain to me like:</b> {p.style}</p>
            {([["You've got", p.knows, "good"], ["Still fuzzy", p.confusions, "warn"], ["You keep coming back to", p.interests, "info"]] as const).map(([t, arr, c]) => arr.length ? (
              <div key={t}><div className="lane-label">{t}</div><div className="tags">{arr.map((x) => <span key={x} className={`tag ${c}`}>{x}</span>)}</div></div>
            ) : null)}
            <div className="small muted">updated {new Date(p.updated).toLocaleString()}</div>
          </>) : <p className="muted">Ask a few questions anywhere in the app and your profile will appear here.</p>}
          <div className="lane-label">Answer style</div>
          <div className="row">
            {(["intuition", "balanced", "math"] as const).map((d) => <button key={d} className={l.depth === d ? "on" : ""} onClick={() => learner.set({ depth: d })}>{d}</button>)}
            <span className="muted small">·</span>
            {(["short", "medium", "long"] as const).map((d) => <button key={d} className={l.length === d ? "on" : ""} onClick={() => learner.set({ length: d })}>{d}</button>)}
          </div>
          <div className="lane-label">My goal</div>
          <textarea className="goal" value={l.goal} rows={3} onChange={(e) => learner.set({ goal: e.target.value })} />
        </div>

        <div className="card">
          <div className="lane-label">Next for you</div>
          {p?.next?.length ? p.next.map((n, i) => (
            <div key={i} className="next-item">
              <b>{n.title}</b><span className="small muted">{n.why}</span>
              {n.try && <a className="small" href="#/xray" onClick={() => ask.set({ tryPrompt: n.try! })}>▶ try “{n.try}”</a>}
            </div>
          )) : <p className="muted small">Appears after a few questions.</p>}

          <div className="lane-label" style={{ marginTop: 18 }}>🛠 Build list — how this app evolves</div>
          <p className="small muted">Ideas collected from your questions, the tutor, and ⭐ buttons. Copy them into a Claude Code session on this repo and the app grows.</p>
          {l.wishes.length === 0 && <p className="muted small">Empty for now.</p>}
          {l.wishes.map((w) => (
            <div key={w.id} className={`wish ${w.done ? "done" : ""}`}>
              <input type="checkbox" checked={!!w.done} onChange={() => learner.set({ wishes: l.wishes.map((x) => x.id === w.id ? { ...x, done: !x.done } : x) })} />
              <span>{w.text}<br /><span className="muted small">from {w.source}</span></span>
              <button className="ghost" onClick={() => learner.set({ wishes: l.wishes.filter((x) => x.id !== w.id) })}>✕</button>
            </div>
          ))}
          <form className="row" onSubmit={(e) => { e.preventDefault(); addWish(wish, "you"); setWish(""); }}>
            <input value={wish} onChange={(e) => setWish(e.target.value)} placeholder="I wish I could see…" style={{ flex: 1 }} />
            <button>Add</button>
          </form>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="primary" onClick={() => navigator.clipboard.writeText(exportPrompt)}>📋 Copy as a request for Claude Code</button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="lane-label">Every question you've asked ({l.history.length})</div>
        {[...l.history].reverse().map((h) => (
          <div key={h.id} className="qa">
            <button className="qa-q" onClick={() => setOpen(open === h.id ? null : h.id)}>
              <span>{h.q}</span><span className="muted small">{h.where} · {new Date(h.ts).toLocaleDateString()}</span>
            </button>
            {open === h.id && <div className="qa-a">{h.selection.length > 0 && <div className="muted small">about: {h.selection.join(" · ")}</div>}<Markdown text={h.a} /></div>}
          </div>
        ))}
        <div className="row" style={{ marginTop: 12 }}>
          <button onClick={() => { const b = new Blob([JSON.stringify(l, null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "topdown-ai-learner.json"; a.click(); }}>⬇ Export my data</button>
          <label className="btn">⬆ Import<input type="file" accept="application/json" hidden onChange={async (e) => { const f = e.target.files?.[0]; if (f) learner.set(JSON.parse(await f.text())); }} /></label>
          <button className="ghost" onClick={() => { if (confirm("Erase your history, profile and build list?")) learner.set({ history: [], profile: null, wishes: [] }); }}>Reset</button>
        </div>
      </div>
    </div>
  );
}
