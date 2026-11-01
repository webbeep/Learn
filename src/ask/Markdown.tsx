import { marked } from "marked";
import { useMemo } from "react";
import { ask } from "../lib/store";

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
marked.use({ renderer: { html: ({ text }: { text: string }) => esc(text) } });

function labelFor(id: string) {
  const el = document.querySelector(`main [data-ask-id="${CSS.escape(id)}"]`);
  const l = el?.getAttribute("data-ask");
  return l ?? id;
}

/** Renders tutor markdown; [[element-id]] → pointer chips, [[try: …]] → run buttons. */
export function Markdown({ text }: { text: string }) {
  const html = useMemo(() => {
    const raw = marked.parse(text, { async: false }) as string;
    return raw.replace(/\[\[\s*try:\s*([^\]]+?)\s*\]\]/gi, (_, p) => `<button class="try-chip" data-try="${p.replace(/"/g, "&quot;")}">▶ try “${p}”</button>`)
      .replace(/\[\[\s*([\w:.-]+)\s*\]\]/g, (_, id) => `<span class="ptr-chip" data-ptr="${id}" title="${esc(labelFor(id))}">◎</span>`);
  }, [text]);
  return (
    <div className="md" dangerouslySetInnerHTML={{ __html: html }}
      onMouseOver={(e) => { const p = (e.target as HTMLElement).closest("[data-ptr]"); if (p) ask.set({ highlight: p.getAttribute("data-ptr") }); }}
      onMouseOut={(e) => { if ((e.target as HTMLElement).closest("[data-ptr]")) ask.set({ highlight: null }); }}
      onClick={(e) => {
        const t = e.target as HTMLElement;
        const p = t.closest("[data-ptr]");
        if (p) {
          const id = p.getAttribute("data-ptr")!;
          document.querySelector(`main [data-ask-id="${CSS.escape(id)}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
          ask.set({ highlight: id }); setTimeout(() => ask.get().highlight === id && ask.set({ highlight: null }), 2500);
        }
        const tr = t.closest("[data-try]");
        if (tr) { ask.set({ tryPrompt: tr.getAttribute("data-try") }); location.hash = "#/xray"; }
      }} />
  );
}
