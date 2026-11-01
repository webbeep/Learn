import { ask, learner, uid, addWish, type AskItem } from "./store";

export interface Turn { role: "user" | "assistant"; content: string; selection?: AskItem[]; followups?: string[]; error?: boolean; via?: "text" | "voice" }

/** Collect every askable element currently on screen (ids the tutor may cite). */
export function visibleElements(limit = 90): AskItem[] {
  const out: AskItem[] = []; const seen = new Set<string>();
  const W = window.innerWidth, H = window.innerHeight;
  document.querySelectorAll<HTMLElement | SVGElement>("main [data-ask-id]").forEach((el) => {
    const r = el.getBoundingClientRect();
    if (r.bottom < 0 || r.top > H || r.right < 0 || r.left > W || (r.width === 0 && r.height === 0)) return; // only what's actually in the window
    const id = el.getAttribute("data-ask-id")!, label = el.getAttribute("data-ask") ?? "";
    if (!seen.has(id) && label) { seen.add(id); out.push({ id, label }); }
  });
  // keep it compact: summarize long runs of cell-level items (e.g. "emb-tok-3:d17")
  if (out.length <= limit) return out;
  const coarse = out.filter((e) => !e.id.includes(":"));
  return coarse.slice(0, limit);
}

export function compressSelection(items: AskItem[], max = 30): AskItem[] {
  if (items.length <= max) return items;
  const groups = new Map<string, AskItem[]>();
  for (const it of items) { const g = it.id.split(":")[0]; groups.set(g, [...(groups.get(g) ?? []), it]); }
  const out: AskItem[] = [];
  for (const [g, its] of groups) {
    if (its.length === 1) out.push(its[0]);
    else out.push({ id: g, label: `${its.length} items, e.g. ${its.slice(0, 4).map((i) => i.label).join("; ")}` });
  }
  return out.slice(0, max);
}

export function viewportNote() {
  const doc = document.documentElement, top = window.scrollY, h = window.innerHeight, total = doc.scrollHeight;
  const heads = [...document.querySelectorAll("main h1, main .lane-label, main .stage-name")].filter((e) => { const r = e.getBoundingClientRect(); return r.top >= 0 && r.top < h; }).slice(0, 6).map((e) => e.textContent?.trim()).filter(Boolean);
  return `window ${window.innerWidth}×${h}px showing ${Math.round((top / total) * 100)}–${Math.round(((top + h) / total) * 100)}% of the page; visible section titles: ${heads.join(" | ") || "none"}`;
}

export async function streamAsk(question: string, thread: Turn[], onText: (t: string) => void, signal?: AbortSignal,
  extra: { mode?: "text" | "voice"; pointer?: { id: string; label: string; ago: number } } = {}) {
  const a = ask.get(), l = learner.get();
  const p = l.profile;
  const body = {
    question,
    thread: thread.filter((t) => !t.error).map(({ role, content }) => ({ role, content })),
    context: { page: a.page, scene: a.scene, summary: a.summary, region: a.region,
      selection: compressSelection(a.selection), elements: visibleElements(), pointer: extra.pointer, viewport: viewportNote() },
    mode: extra.mode ?? "text",
    learner: { goal: l.goal, depth: l.depth, length: l.length,
      profile: p ? `${p.summary} Style: ${p.style}. Confusions: ${p.confusions.join("; ")}` : undefined,
      recent: l.history.slice(-5).map((h) => h.q) },
  };
  const res = await fetch("/api/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal });
  if (!res.ok || !res.body) throw new Error(await res.text() || `HTTP ${res.status}`);
  const reader = res.body.getReader(), dec = new TextDecoder();
  let full = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    full += dec.decode(value, { stream: true }); onText(full);
  }
  return full;
}

/** Split tutor output into body, follow-up suggestions and build requests. */
export function parseAnswer(text: string) {
  const followups: string[] = []; const builds: string[] = [];
  const body = text.split("\n").filter((line) => {
    const f = line.match(/^\s*\**FOLLOWUPS:?\**:?\s*(.*)$/i);
    if (f) { followups.push(...f[1].split("|").map((s) => s.trim()).filter(Boolean)); return false; }
    const b = line.match(/^\s*\**BUILD:?\**:?\s*(.*)$/i);
    if (b) { builds.push(b[1].trim()); return false; }
    return true;
  }).join("\n").trim();
  return { body, followups, builds };
}

export function recordQA(question: string, answer: string, selection: AskItem[]) {
  const a = ask.get();
  const { body, builds } = parseAnswer(answer);
  builds.forEach((b) => addWish(b, `tutor, while you asked: "${question.slice(0, 80)}"`));
  learner.set((s) => ({ history: [...s.history, { id: uid(), ts: Date.now(), where: `${a.page}/${a.scene}`,
    selection: selection.map((x) => x.label).slice(0, 6), q: question, a: body }].slice(-200) }));
  const n = learner.get().history.length;
  if (n === 1 || n % 3 === 0) refreshProfile().catch(() => {});
}

let refreshing = false;
export async function refreshProfile() {
  if (refreshing) return; refreshing = true;
  try {
    const l = learner.get();
    const res = await fetch("/api/profile", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ previous: l.profile, history: l.history.slice(-15).map(({ q, a, where }) => ({ q, a, where })) }) });
    if (!res.ok) throw new Error(await res.text());
    const p = await res.json();
    const arr = (x: unknown) => (Array.isArray(x) ? x : []);
    learner.set({ profile: { summary: p.summary ?? "", knows: arr(p.knows), confusions: arr(p.confusions), interests: arr(p.interests),
      style: p.style ?? "", next: arr(p.next), build: arr(p.build), updated: Date.now() } });
    arr(p.build).forEach((b: string) => addWish(b, "learner profile"));
  } finally { refreshing = false; }
}
