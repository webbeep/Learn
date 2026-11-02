import { useEffect, useRef, useState } from "react";
import { ask, select, useStore, type AskItem } from "../lib/store";
import { afterHighlight } from "../lib/conversation";

/** Click-to-select + drag-to-highlight layer for everything inside <main>. */
export function useClickToAsk(main: React.RefObject<HTMLElement | null>) {
  useEffect(() => {
    const el = main.current; if (!el) return;
    const onClick = (e: MouseEvent) => {
      if (ask.get().regionMode) return;
      const t = (e.target as Element).closest("[data-ask-id]");
      if (!t || (e.target as Element).closest("button, input, select, textarea, a")) return;
      const item = { id: t.getAttribute("data-ask-id")!, label: t.getAttribute("data-ask") ?? "" };
      const cur = ask.get().selection;
      const items = e.shiftKey || e.metaKey ? (cur.some((c) => c.id === item.id) ? cur.filter((c) => c.id !== item.id) : [...cur, item]) : [item];
      select(items, "", false);
      if (!ask.get().open) window.dispatchEvent(new CustomEvent("ask-popover", { detail: { x: e.clientX, y: e.clientY } }));
    };
    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  }, [main]);
}

export function RegionLayer({ main }: { main: React.RefObject<HTMLElement | null> }) {
  const on = useStore(ask, (s) => s.regionMode);
  const [rect, setRect] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const start = useRef<{ x: number; y: number } | null>(null);
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") ask.set({ regionMode: false }); };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  }, []);
  if (!on) return null;
  const finish = () => {
    const r = rect; start.current = null; setRect(null);
    if (!r || !main.current) return;
    const L = Math.min(r.x0, r.x1), R = Math.max(r.x0, r.x1), T = Math.min(r.y0, r.y1), B = Math.max(r.y0, r.y1);
    if (R - L < 6 || B - T < 6) return;
    const items: AskItem[] = []; const seen = new Set<string>();
    main.current.querySelectorAll("[data-ask-id]").forEach((n) => {
      const b = n.getBoundingClientRect();
      const cx = (b.left + b.right) / 2, cy = (b.top + b.bottom) / 2;
      const id = n.getAttribute("data-ask-id")!;
      if (cx >= L && cx <= R && cy >= T && cy <= B && !seen.has(id)) { seen.add(id); items.push({ id, label: n.getAttribute("data-ask") ?? "" }); }
    });
    const m = main.current.getBoundingClientRect();
    const where = `${Math.round(((L + R) / 2 - m.left) / m.width * 100)}% across, ${Math.round(((T + B) / 2 - m.top) / m.height * 100)}% down the stage`;
    select(items, `${items.length} element(s) inside a box ${Math.round(R - L)}×${Math.round(B - T)}px, centered ${where}`, false);
    ask.set({ regionMode: false });
    afterHighlight({ x: (L + R) / 2, y: B });
  };
  return (
    <div className="fixed inset-0 z-40 cursor-crosshair touch-none bg-black/20"
      onPointerDown={(e) => { start.current = { x: e.clientX, y: e.clientY }; setRect({ x0: e.clientX, y0: e.clientY, x1: e.clientX, y1: e.clientY }); (e.target as Element).setPointerCapture(e.pointerId); }}
      onPointerMove={(e) => start.current && setRect({ x0: start.current.x, y0: start.current.y, x1: e.clientX, y1: e.clientY })}
      onPointerUp={finish}>
      <div className="pointer-events-none fixed top-16 left-1/2 -translate-x-1/2 rounded-full bg-pin px-4 py-1.5 text-sm font-medium text-black shadow-lg">Drag a box over anything you're curious about · Esc to cancel</div>
      {rect && <div className="pointer-events-none fixed rounded-md border-2 border-dashed border-pin bg-pin/10" style={{ left: Math.min(rect.x0, rect.x1), top: Math.min(rect.y0, rect.y1), width: Math.abs(rect.x1 - rect.x0), height: Math.abs(rect.y1 - rect.y0) }} />}
    </div>
  );
}

/** Injects CSS that highlights selected / pointed-at elements, surviving re-renders. */
export function MarkStyles() {
  const sel = useStore(ask, (s) => s.selection);
  const hi = useStore(ask, (s) => s.highlight);
  const q = (id: string) => `main [data-ask-id="${CSS.escape(id)}"]`;
  const css = [
    sel.length ? `${sel.map((s) => q(s.id)).join(",")}{outline:2px solid var(--viz-sel);outline-offset:1px;border-radius:4px;stroke:var(--viz-sel)!important;stroke-width:2px;}` : "",
    hi ? `${q(hi)}{outline:3px solid var(--viz-pin);outline-offset:2px;border-radius:4px;stroke:var(--viz-pin)!important;stroke-width:3px;filter:drop-shadow(0 0 6px var(--viz-pin));animation:pulse-pin 0.9s ease-in-out infinite alternate}` : "",
    "main svg [data-ask-id]{outline:none!important}",
    "main svg [data-ask-id] text{stroke:none!important}",
  ].join("\n");
  return <style>{css}</style>;
}
