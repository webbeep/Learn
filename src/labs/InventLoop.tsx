// The research loop as an interactive stepper, with tiny visuals and real numbers where we have them.
import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, Beaker, FlaskConical, GitCompare, Lightbulb, Minimize2, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useResearch } from "./charts";

interface Ablations { steps: number; runs: { id: string; label: string; params: number; final_val: number | null }[] }

// Measured by running this page's exported train.py (character-level, 400 steps, batch 32, CPU) with seeds 0, 1, 2.
export const SEED_RUNS = { steps: 400, baseline: [0.917, 0.9793, 0.9794], nopos: [1.3422, 1.3343, 1.345] };

const STEPS = [
  { id: "hypothesis", icon: Lightbulb, title: "Hypothesis", short: "Guess" },
  { id: "small", icon: Minimize2, title: "Smallest experiment", short: "Shrink" },
  { id: "baseline", icon: Beaker, title: "Baseline", short: "Baseline" },
  { id: "change", icon: FlaskConical, title: "Change one thing", short: "Change" },
  { id: "seeds", icon: GitCompare, title: "Compare across seeds", short: "Seeds" },
  { id: "scale", icon: TrendingUp, title: "Scale up", short: "Scale" },
] as const;

export function ResearchLoop({ onSummary }: { onSummary: (s: string) => void }) {
  const [i, setI] = useState(0);
  const abl = useResearch<Ablations>("/research/ablations.json").data;
  const base = abl?.runs.find((r) => r.id === "baseline"), nopos = abl?.runs.find((r) => r.id === "no-positions");
  const s = STEPS[i];
  const body = BODY(abl, base, nopos)[s.id];
  useEffect(() => { onSummary(`Research loop stepper, step ${i + 1}/6 "${s.title}": ${body.text}`); }, [i, s, body.text, onSummary]);

  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle>The research loop</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-6 gap-1.5" role="tablist">
          {STEPS.map((st, j) => (
            <button key={st.id} role="tab" aria-selected={i === j} onClick={() => setI(j)}
              className={cn("flex min-h-12 flex-col items-center justify-center gap-1 rounded-lg border px-1 py-1.5 text-xs transition-colors @2xl:flex-row @2xl:gap-2 @2xl:text-sm",
                i === j ? "border-foreground/30 bg-muted font-medium" : "text-muted-foreground hover:bg-muted/50", j < i && "text-foreground")}>
              <span className={cn("flex size-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold", i === j ? "bg-primary text-primary-foreground" : j < i ? "bg-viz-a text-background" : "bg-muted")}>{j + 1}</span>
              <span className="hidden @md:inline @2xl:hidden">{st.short}</span><span className="hidden @2xl:inline">{st.title}</span>
            </button>
          ))}
        </div>
        <div className="grid grid-cols-1 items-center gap-5 rounded-xl border bg-[var(--viz-stage)] p-4 @2xl:grid-cols-5 [&>*]:min-w-0">
          <div className="@2xl:col-span-2" data-ask-id={`loop-visual-${s.id}`} data-ask={`research loop step "${s.title}" illustration: ${body.text}`}>{body.visual}</div>
          <div className="space-y-2 @2xl:col-span-3">
            <div className="flex items-center gap-2"><s.icon className="size-5 text-pin" /><h3 className="text-lg font-semibold">{i + 1}. {s.title}</h3></div>
            <p className="text-pretty">{body.text}</p>
            <div className="rounded-lg border bg-card p-3 text-sm"><span className="font-medium">Example: </span><span className="text-muted-foreground">{body.example}</span></div>
          </div>
        </div>
        <div className="flex justify-between">
          <Button variant="outline" className="h-9" disabled={i === 0} onClick={() => setI(i - 1)}><ArrowLeft />Back</Button>
          <Button className="h-9" onClick={() => setI((i + 1) % STEPS.length)}>{i === STEPS.length - 1 ? "Start over" : "Next"}<ArrowRight /></Button>
        </div>
      </CardContent>
    </Card>
  );
}

type Run = Ablations["runs"][number] | undefined;
const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;

function BODY(abl: Ablations | null, base: Run, nopos: Run): Record<(typeof STEPS)[number]["id"], { text: string; example: React.ReactNode; visual: React.ReactNode }> {
  const bv = base?.final_val, nv = nopos?.final_val;
  return {
    hypothesis: {
      text: "Write down what you expect and why, before running anything. A good hypothesis predicts a direction and roughly how big the effect is, so the experiment can prove you wrong.",
      example: "\"TinyGPT needs its position embeddings to copy the right color: without them, val loss will be clearly worse.\"",
      visual: <HypothesisViz />,
    },
    small: {
      text: "Shrink the experiment until it runs in minutes. You learn from the number of experiments you finish, not the size of each one.",
      example: "TinyGPT: 170,560 parameters, 80 seconds on a laptop CPU. Short runs show big effects fast, but some skills (like copying the right color) only appear later in training, so check more than one budget.",
      visual: <ShrinkViz />,
    },
    baseline: {
      text: "Run the unchanged recipe with exactly the same data, steps and seed. Without a baseline, a number means nothing.",
      example: bv != null ? <>Baseline TinyGPT, {abl!.steps.toLocaleString()} steps: val loss <b className="text-foreground">{bv.toFixed(3)}</b> (real run from the Architecture lab experiments).</> : "Train the unchanged TinyGPT for 1,500 steps and record its val loss.",
      visual: <BarsViz bars={[{ label: "baseline", v: bv ?? 0.49, color: "var(--muted-foreground)" }]} max={Math.max(bv ?? 0.5, nv ?? 0.6) * 1.15} real={bv != null} />,
    },
    change: {
      text: "Change exactly one thing. If you change two, you won't know which one mattered.",
      example: nv != null && bv != null ? (Math.abs(nv / bv - 1) < 0.02
        ? <>positions: learned → none. Val loss <b className="text-foreground">{bv.toFixed(3)} → {nv.toFixed(3)}</b> ({((nv / bv - 1) * 100).toFixed(1)}%). Almost no difference: <b className="text-foreground">the hypothesis fails</b>. That's a win for science: with a causal mask, a token can work out roughly where it is from what came before it, so the model found another way. A failed prediction tells you more than a confirmed one.</>
        : <>positions: learned → none. Val loss <b className="text-foreground">{bv.toFixed(3)} → {nv.toFixed(3)}</b> ({((nv / bv - 1) * 100).toFixed(0)}% worse). Hypothesis survives, for now.</>) : "positions: learned → none. Everything else identical.",
      visual: <div className="space-y-3"><DiffViz /><BarsViz bars={[{ label: "baseline", v: bv ?? 0.49, color: "var(--muted-foreground)" }, { label: "no positions", v: nv ?? 0.61, color: "var(--viz-pin)" }]} max={Math.max(bv ?? 0.5, nv ?? 0.6) * 1.15} real={bv != null && nv != null} /></div>,
    },
    seeds: {
      text: "Random init and batch order add noise. Rerun with 3+ seeds: an effect only counts if it's bigger than the spread between seeds.",
      example: <>With the script this page exports (character-level, {SEED_RUNS.steps} steps): baseline {mean(SEED_RUNS.baseline).toFixed(3)} ± {spread(SEED_RUNS.baseline)}, no positions {mean(SEED_RUNS.nopos).toFixed(3)} ± {spread(SEED_RUNS.nopos)}. The gap is ~{Math.round((mean(SEED_RUNS.nopos) - mean(SEED_RUNS.baseline)) / Math.max(...[SEED_RUNS.baseline, SEED_RUNS.nopos].map((a) => Math.max(...a) - Math.min(...a))))}× the seed spread: real. Notice this short character-level run says positions matter, while the full-length TinyGPT run above says they barely do. Same question, different setup and budget, different answer. That's why you state the setup with every result.</>,
      visual: <SeedsViz />,
    },
    scale: {
      text: "Does the win survive at 10× the size? Many tricks help tiny models and vanish at scale (or the reverse). Check the trend across sizes, not one point.",
      example: <>Re-run baseline and variant at 3+ sizes and plot both lines, like the <a href="#/scaling" className="underline">Scaling lab</a>. Parallel lines = a constant win; converging lines = it fades with scale.</>,
      visual: <ScaleViz />,
    },
  };
}
const spread = (a: number[]) => ((Math.max(...a) - Math.min(...a)) / 2).toFixed(3);

// ---------- small visuals --------------------------------------------------------
const VB = "0 0 260 150";
function HypothesisViz() {
  return (
    <svg viewBox={VB} className="mx-auto w-full max-w-72">
      <rect x={20} y={28} width={220} height={94} rx={12} fill="var(--card)" stroke="var(--border)" />
      <text x={36} y={56} fontSize={12} className="fill-muted-foreground">if</text>
      <rect x={54} y={43} width={160} height={18} rx={5} fill="color-mix(in oklch, var(--viz-a) 25%, transparent)" />
      <text x={60} y={56} fontSize={11.5} className="fill-foreground font-mono">positions removed</text>
      <text x={36} y={84} fontSize={12} className="fill-muted-foreground">then</text>
      <rect x={66} y={71} width={140} height={18} rx={5} fill="color-mix(in oklch, var(--viz-pin) 30%, transparent)" />
      <text x={72} y={84} fontSize={11.5} className="fill-foreground font-mono">val loss ↑ &gt;10%</text>
      <text x={36} y={110} fontSize={12} className="fill-muted-foreground">because order matters</text>
    </svg>
  );
}
function ShrinkViz() {
  return (
    <svg viewBox={VB} className="mx-auto w-full max-w-72">
      <rect x={14} y={20} width={110} height={110} rx={10} fill="none" stroke="var(--muted-foreground)" strokeDasharray="4 4" />
      <text x={69} y={78} fontSize={11} textAnchor="middle" className="fill-muted-foreground">weeks on GPUs</text>
      <path d="M134 75 L170 75" stroke="var(--foreground)" strokeWidth={2} markerEnd="url(#arr)" />
      <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--foreground)" /></marker></defs>
      <rect x={182} y={55} width={44} height={44} rx={8} fill="color-mix(in oklch, var(--viz-a) 35%, transparent)" stroke="var(--viz-a)" />
      <text x={204} y={81} fontSize={10.5} textAnchor="middle" className="fill-foreground font-semibold">80 s</text>
      <text x={204} y={118} fontSize={10.5} textAnchor="middle" className="fill-muted-foreground">TinyGPT</text>
    </svg>
  );
}
function BarsViz({ bars, max, real }: { bars: { label: string; v: number; color: string }[]; max: number; real: boolean }) {
  return (
    <div className="space-y-2">
      {bars.map((b) => (
        <div key={b.label} className="space-y-1">
          <div className="flex justify-between text-xs"><span className="text-muted-foreground">{b.label}</span><span className="font-mono tabular-nums">{b.v.toFixed(3)}</span></div>
          <div className="h-4 rounded-r-md" style={{ width: `${(b.v / max) * 100}%`, background: b.color }} />
        </div>
      ))}
      <div className="text-[11px] text-muted-foreground">val loss, lower is better{real ? " · real runs" : " · loading real numbers…"}</div>
    </div>
  );
}
function DiffViz() {
  const rows = [["n_layer", "3", ""], ["d_model", "64", ""], ["positions", "learned", "none"], ["lr", "3e-3", ""]];
  return (
    <div className="overflow-hidden rounded-lg border bg-card font-mono text-xs">
      {rows.map(([k, a, b]) => (
        <div key={k} className={cn("flex gap-2 px-2.5 py-1", b && "bg-[color-mix(in_oklch,var(--viz-pin)_18%,transparent)]")}>
          <span className="w-20 text-muted-foreground">{k}</span>{b ? <><span className="line-through opacity-60">{a}</span><span>→ {b}</span><Badge variant="outline" className="ml-auto h-4 px-1 text-[10px]">the one change</Badge></> : <span>{a}</span>}
        </div>
      ))}
    </div>
  );
}
function SeedsViz() {
  const all = [...SEED_RUNS.baseline, ...SEED_RUNS.nopos], lo = Math.min(...all) - 0.05, hi = Math.max(...all) + 0.05;
  const X = (v: number) => 20 + ((v - lo) / (hi - lo)) * 220;
  const rows: [string, number[], string][] = [["baseline", SEED_RUNS.baseline, "var(--muted-foreground)"], ["no positions", SEED_RUNS.nopos, "var(--viz-pin)"]];
  return (
    <svg viewBox={VB} className="mx-auto w-full max-w-72">
      {rows.map(([label, vals, c], r) => (
        <g key={label}>
          <text x={20} y={30 + r * 55} fontSize={11} className="fill-muted-foreground">{label}</text>
          <line x1={X(Math.min(...vals))} x2={X(Math.max(...vals))} y1={48 + r * 55} y2={48 + r * 55} stroke={c} strokeWidth={2} />
          {vals.map((v, k) => <circle key={k} cx={X(v)} cy={48 + r * 55} r={5} fill={c} stroke="var(--background)" strokeWidth={1.5}><title>{`seed ${k}: ${v}`}</title></circle>)}
          <text x={X(mean(vals))} y={66 + r * 55} fontSize={10} textAnchor="middle" className="fill-foreground font-mono">{mean(vals).toFixed(3)}</text>
        </g>
      ))}
      <text x={130} y={144} fontSize={10} textAnchor="middle" className="fill-muted-foreground">val loss, 3 seeds each (real runs)</text>
    </svg>
  );
}
function ScaleViz() {
  const xs = [0, 1, 2, 3], base = [1.6, 1.15, 0.85, 0.7], v1 = [1.45, 1.02, 0.74, 0.6], v2 = [1.4, 1.08, 0.84, 0.71];
  const X = (i: number) => 30 + i * 62, Y = (v: number) => 20 + (1.7 - v) * 85;
  const line = (a: number[]) => a.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join("");
  return (
    <svg viewBox={VB} className="mx-auto w-full max-w-72">
      <path d={line(base)} fill="none" stroke="var(--muted-foreground)" strokeWidth={2} />
      <path d={line(v1)} fill="none" stroke="var(--viz-b)" strokeWidth={2} />
      <path d={line(v2)} fill="none" stroke="var(--viz-pin)" strokeWidth={2} strokeDasharray="4 3" />
      {xs.map((i) => <text key={i} x={X(i)} y={142} fontSize={9.5} textAnchor="middle" className="fill-muted-foreground">{["10k", "100k", "1M", "10M"][i]}</text>)}
      <text x={X(3) + 4} y={Y(0.6) + 3} fontSize={9.5} className="fill-foreground">keeps</text>
      <text x={X(3) + 4} y={Y(0.74) + 4} fontSize={9.5} className="fill-foreground">fades</text>
      <text x={34} y={14} fontSize={9.5} className="fill-muted-foreground">illustration, not data</text>
    </svg>
  );
}
