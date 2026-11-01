import { useEffect, useRef, useState } from "react";
import { loadModel, type TinyGPT } from "./model/gpt";
import { ask, learner, markVisited, useStore } from "./lib/store";
import { AskPanel, AskPopover } from "./ask/AskPanel";
import { MarkStyles, RegionLayer, useClickToAsk } from "./ask/Interact";
import { XRay } from "./pages/XRay";
import { Training } from "./pages/Training";
import { Watch } from "./pages/Watch";
import { MapPage } from "./pages/MapPage";
import { Path } from "./pages/Path";

const ROUTES = [
  { href: "#/", label: "Journey", icon: "🗺️" },
  { href: "#/xray", label: "X-ray", icon: "🔬" },
  { href: "#/training", label: "Training", icon: "📉" },
  { href: "#/watch", label: "Watch", icon: "🎬" },
  { href: "#/path", label: "My path", icon: "🧭" },
];

function useHash() {
  const [h, setH] = useState(location.hash || "#/");
  useEffect(() => { const f = () => setH(location.hash || "#/"); window.addEventListener("hashchange", f); return () => window.removeEventListener("hashchange", f); }, []);
  return h;
}

export default function App() {
  const hash = useHash();
  const [model, setModel] = useState<TinyGPT | null>(null);
  const [err, setErr] = useState("");
  const main = useRef<HTMLElement>(null);
  const panelOpen = useStore(ask, (s) => s.open);
  const nWishes = useStore(learner, (s) => s.wishes.filter((w) => !w.done).length);
  useClickToAsk(main);
  useEffect(() => { loadModel().then(setModel).catch((e) => setErr(String(e))); }, []);
  useEffect(() => { markVisited(hash); ask.set({ selection: [], region: "", highlight: null }); window.scrollTo(0, 0); }, [hash]);

  let page: React.ReactNode;
  if (hash.startsWith("#/watch")) page = <Watch />;
  else if (hash.startsWith("#/path")) page = <Path />;
  else if (hash.startsWith("#/xray") || hash.startsWith("#/training")) {
    page = err ? <div className="loading error">Couldn't load the model: {err}</div> : !model ? <div className="loading">Loading TinyGPT (680 KB of weights)…</div>
      : hash.startsWith("#/xray") ? <XRay m={model} /> : <Training corpus={model.meta.corpus_sample} />;
  } else page = <MapPage />;

  return (
    <div className={`app ${panelOpen ? "with-panel" : ""}`}>
      <header className="topbar">
        <a href="#/" className="brand">🔭 <b>Top-Down</b> AI</a>
        <nav>
          {ROUTES.map((r) => (
            <a key={r.href} href={r.href} className={(r.href === "#/" ? hash === "#/" || hash === "" : hash.startsWith(r.href)) ? "on" : ""}>
              <span>{r.icon}</span> <span className="nav-label">{r.label}</span>{r.href === "#/path" && nWishes > 0 && <sup>{nWishes}</sup>}
            </a>
          ))}
        </nav>
        <button className="primary" onClick={() => ask.set({ open: !panelOpen })}>💬 <span className="nav-label">Ask</span></button>
      </header>
      <main ref={main}>{page}</main>
      <AskPanel />
      <AskPopover />
      <RegionLayer main={main} />
      <MarkStyles />
    </div>
  );
}
