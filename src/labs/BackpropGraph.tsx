// Interactive computational graph: step forward, step backward, nudge, take gradient steps.
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, FastForward, Pause, Play, RotateCcw, TrendingDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { div } from "../lib/viz";
import type { Value } from "./autograd";
import { OP_NAME, PRESETS, backwardText, fmt, forwardText, layout, localRule, simulate, type Preset } from "./graphs";
import { useWidth } from "./charts";

const init = (p: Preset) => Object.fromEntries(p.leaves.map((l) => [l.key, l.init]));

export function GraphLab({ onSummary }: { onSummary: (scene: string, s: string) => void }) {
  const [pid, setPid] = useState("neuron");
  const preset = PRESETS.find((p) => p.id === pid)!;
  const [vals, setVals] = useState<Record<string, number>>(() => init(preset));
  const [fwd, setFwd] = useState(0);
  const [bwd, setBwd] = useState(0);
  const [sel, setSel] = useState<string | null>(null);
  const [lr, setLr] = useState(0.1);
  const [hist, setHist] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);

  const root = useMemo(() => preset.build(vals), [preset, vals]);
  const g = useMemo(() => layout(root), [root]);
  const sim = useMemo(() => simulate(root, g.topo), [root, g]);
  const nOps = sim.ops.length, nB = nOps + 1;
  const live = fwd === nOps && bwd === nB;
  const byLabel = useMemo(() => new Map(g.topo.map((v) => [v.label, v])), [g]);

  const switchPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setPid(id); setVals(init(p)); setFwd(0); setBwd(0); setSel(null); setHist([]); setPlaying(false);
  };
  const stepF = () => { if (fwd < nOps) setFwd(fwd + 1); else if (bwd < nB) setBwd(bwd + 1); };
  const stepB = () => { if (bwd > 0) setBwd(bwd - 1); else if (fwd > 0) setFwd(fwd - 1); };
  const goLive = () => { setFwd(nOps); setBwd(nB); };
  const reset = () => { setVals(init(preset)); setFwd(0); setBwd(0); setHist([]); setPlaying(false); };

  // autoplay: one node per tick
  const stepRef = useRef(stepF); stepRef.current = stepF;
  useEffect(() => {
    if (!playing) return;
    if (live) { setPlaying(false); return; }
    const t = setTimeout(() => stepRef.current(), fwd < nOps ? 650 : 900);
    return () => clearTimeout(t);
  }, [playing, fwd, bwd, live, nOps]);

  const nudge = (k: string, v: number) => { setVals((s) => ({ ...s, [k]: v })); goLive(); };
  const gradStep = (n = 1) => {
    let cur = { ...vals };
    const losses: number[] = [];
    for (let i = 0; i < n; i++) {
      const r = preset.build(cur); r.backward();
      if (!hist.length && !losses.length) losses.push(r.data);
      const leaves = new Map(r.topo().filter((v) => !v.prev.length).map((v) => [v.label, v]));
      cur = Object.fromEntries(preset.leaves.map((l) => {
        const lv = leaves.get(l.key)!;
        const nv = l.kind === "param" ? lv.data - lr * lv.grad : lv.data;
        return [l.key, Math.max(l.min * 3, Math.min(l.max * 3, nv))];
      }));
      losses.push(preset.build(cur).data);
    }
    setVals(cur); setHist((h) => [...h, ...losses].slice(-200)); goLive();
  };

  // what's happening right now
  const lastF = fwd > 0 && bwd === 0 ? sim.ops[fwd - 1] : null;
  const lastB = bwd >= 2 ? sim.ops[nOps - (bwd - 1)] : null;
  const grads = sim.grads[bwd], reached = sim.reached[bwd];
  const known = (v: Value) => !v.prev.length || sim.ops.indexOf(v) < fwd;
  const gScale = Math.max(1e-9, ...g.topo.filter((v) => reached.has(v.label)).map((v) => Math.abs(grads[v.label])));
  const phase = bwd > 0 ? "backward" : fwd > 0 ? "forward" : "start";

  let caption: React.ReactNode;
  if (live) caption = <>All values and gradients are computed. <b>Drag any slider</b> and everything updates instantly, or <b>take a gradient step</b> below.</>;
  else if (phase === "start") caption = <>Leaves (left) are the numbers we choose. Press <b>Forward</b> to compute each node from its inputs, one at a time.</>;
  else if (lastF) caption = <span className="font-mono text-[13px]">{forwardText(lastF)}</span>;
  else if (bwd === 1) caption = <>Backward starts at the end: <span className="font-mono text-[13px]">∂{preset.lossLabel}/∂{preset.lossLabel} = 1</span>. Nudging the loss by ε changes the loss by ε.</>;
  else if (lastB) caption = (
    <div className="space-y-0.5">
      <div className="text-xs text-muted-foreground">Chain rule at <b className="font-mono text-foreground">{lastB.label}</b>: local slope × gradient arriving from the right</div>
      {backwardText(lastB, preset.lossLabel).map((t) => <div key={t} className="font-mono text-[12.5px] break-words">{t}</div>)}
    </div>
  );

  useEffect(() => {
    const leaves = preset.leaves.map((l) => `${l.key}=${fmt(vals[l.key])}${live ? ` (grad ${fmt(sim.grads[nB][l.key])})` : ""}`).join(", ");
    onSummary(`Graph: ${preset.name}`, [
      `Computational graph playground, preset "${preset.name}": ${preset.formula}.`,
      `Leaf values: ${leaves}. Output ${preset.lossLabel} = ${fmt(root.data, 4)}.`,
      `State: forward ${fwd}/${nOps} nodes computed, backward ${bwd}/${nB} steps.`,
      lastF ? `Just computed: ${forwardText(lastF)}.` : "",
      lastB ? `Just back-propagated through ${lastB.label}: ${backwardText(lastB, preset.lossLabel).join("; ")}.` : "",
      hist.length ? `Gradient steps taken: ${hist.length - 1}, loss ${fmt(hist[0], 4)} → ${fmt(hist[hist.length - 1], 4)} with learning rate ${lr}.` : "",
    ].filter(Boolean).join("\n"));
  }, [preset, vals, fwd, bwd, root, hist, lr, live, lastF, lastB, nOps, nB, sim, onSummary]);

  const selV = sel ? byLabel.get(sel) ?? null : null;

  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle>The chain rule, one node at a time</CardTitle>
        <CardDescription>Forward computes values left → right. Backward sends gradients right → left: how much the {preset.lossLabel} would change if you nudged each number.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col gap-3 @2xl:flex-row @2xl:items-center @2xl:justify-between">
          <ToggleGroup type="single" variant="outline" value={pid} onValueChange={(v) => v && switchPreset(v)} className="w-full @2xl:w-auto">
            {PRESETS.map((p) => <ToggleGroupItem key={p.id} value={p.id} className="h-9 flex-1 px-3 @2xl:flex-none">{p.name}</ToggleGroupItem>)}
          </ToggleGroup>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="outline" className="h-9" onClick={stepB} disabled={fwd === 0 && bwd === 0} aria-label="Step back"><ArrowLeft /></Button>
            <Button className="h-9 min-w-32 flex-1 @2xl:flex-none" onClick={stepF} disabled={live}>
              {fwd < nOps ? <>Forward <span className="tabular-nums opacity-70">{fwd}/{nOps}</span></> : <>Backward <span className="tabular-nums opacity-70">{bwd}/{nB}</span></>}<ArrowRight />
            </Button>
            <Button variant="outline" className="h-9" onClick={() => setPlaying(!playing)} disabled={live} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause /> : <Play />}</Button>
            <Button variant="outline" className="h-9" onClick={goLive} disabled={live} title="Compute everything"><FastForward /></Button>
            <Button variant="ghost" className="h-9" onClick={reset} title="Reset"><RotateCcw /></Button>
          </div>
        </div>

        <PhaseBar fwd={fwd} nOps={nOps} bwd={bwd} nB={nB} />
        <div className="-mb-1 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm"><span className="text-muted-foreground">{preset.blurb}</span><span className="font-mono text-[13px]">{preset.formula}</span></div>

        <Graph g={g} grads={grads} reached={reached} known={known} gScale={gScale} lastF={lastF} lastB={lastB} rootLive={bwd === 1}
          sel={sel} onSel={(l) => setSel(sel === l ? null : l)} lossLabel={preset.lossLabel} leafKind={(k) => preset.leaves.find((l) => l.key === k)?.kind} />

        <div className="min-h-14 rounded-lg border bg-muted/40 px-3 py-2.5 text-sm" aria-live="polite">{caption}</div>

        {selV && <NodeDetail v={selV} preset={preset} vals={vals} live={live} grad={grads[selV.label]} hasGrad={reached.has(selV.label)} known={known(selV)} onClose={() => setSel(null)} />}

        <div className="grid grid-cols-1 gap-4 @3xl:grid-cols-5 [&>*]:min-w-0">
          <div className="space-y-1 @3xl:col-span-3">
            <div className="mb-2 flex items-baseline justify-between"><div className="text-sm font-medium">Nudge the leaves</div><div className="text-xs text-muted-foreground">gradient shown once computed</div></div>
            <div className="grid gap-x-6 gap-y-1 @xl:grid-cols-2">
              {preset.leaves.map((l) => {
                const gr = sim.grads[bwd][l.key], has = reached.has(l.key);
                return (
                  <div key={l.key} className={cn("rounded-lg px-2 py-1.5", sel === l.key && "bg-muted")} data-ask-id={`bp-leaf-${l.key}`}
                    data-ask={`leaf ${l.key} (${l.hint}) = ${fmt(vals[l.key])}${has ? `, gradient ∂${preset.lossLabel}/∂${l.key} = ${fmt(gr, 3)}` : ""}`}>
                    <div className="flex items-center gap-2 text-sm">
                      <button className="font-mono font-semibold hover:underline" onClick={() => setSel(l.key)}>{l.key}</button>
                      <span className="truncate text-xs text-muted-foreground">{l.kind === "param" ? "learned" : l.kind}</span>
                      <span className="ml-auto font-mono tabular-nums">{fmt(vals[l.key])}</span>
                      <span className="w-20 text-right font-mono text-xs tabular-nums" style={{ color: has ? gradInk(gr) : undefined }}>{has ? `∇ ${fmt(gr)}` : "∇ ?"}</span>
                    </div>
                    <Slider className="py-2" min={l.min} max={l.max} step={0.01} value={[vals[l.key]]} onValueChange={([v]) => nudge(l.key, v)} aria-label={l.key} />
                  </div>
                );
              })}
            </div>
          </div>
          <div className="space-y-3 rounded-xl border p-3 @3xl:col-span-2">
            <div className="flex items-center gap-2 text-sm font-medium"><TrendingDown className="size-4" />Gradient descent</div>
            <p className="text-xs text-muted-foreground">Move every <i>learned</i> number a little against its gradient: <span className="font-mono">w ← w − lr · ∇w</span>. Inputs and targets stay fixed.</p>
            <div className="flex items-center gap-3 text-sm"><span className="w-24 shrink-0 text-muted-foreground">learning rate</span>
              <Slider min={0.01} max={pid === "expr" ? 0.1 : 1} step={0.01} value={[Math.min(lr, pid === "expr" ? 0.1 : 1)]} onValueChange={([v]) => setLr(v)} />
              <span className="w-10 text-right font-mono tabular-nums">{Math.min(lr, pid === "expr" ? 0.1 : 1).toFixed(2)}</span></div>
            <div className="flex gap-2">
              <Button className="h-9 flex-1" onClick={() => gradStep(1)}>Take a gradient step</Button>
              <Button variant="outline" className="h-9" onClick={() => gradStep(10)}>×10</Button>
            </div>
            <Sparkline values={hist} label={preset.lossLabel} />
          </div>
        </div>
      </CardContent>
      <style>{`
        @keyframes bp-flow { to { stroke-dashoffset: -28; } }
        @keyframes bp-flow-back { to { stroke-dashoffset: 28; } }
        .bp-fwd { stroke-dasharray: 6 8; animation: bp-flow .7s linear infinite; }
        .bp-bwd { stroke-dasharray: 6 8; animation: bp-flow-back .7s linear infinite; }
        @media (prefers-reduced-motion: reduce) { .bp-fwd, .bp-bwd { animation: none; } }
      `}</style>
    </Card>
  );
}

const gradInk = (g: number) => (g === 0 ? "var(--muted-foreground)" : g > 0 ? "var(--grad-pos)" : "var(--grad-neg)");

function PhaseBar({ fwd, nOps, bwd, nB }: { fwd: number; nOps: number; bwd: number; nB: number }) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className="w-14 shrink-0">forward</span>
      <div className="flex flex-1 gap-0.5">{Array.from({ length: nOps }, (_, i) => <span key={i} className={cn("h-1.5 flex-1 rounded-full", i < fwd ? "bg-viz-a" : "bg-muted")} />)}</div>
      <span className="w-16 shrink-0 text-right">backward</span>
      <div className="flex flex-1 gap-0.5">{Array.from({ length: nB }, (_, i) => <span key={i} className={cn("h-1.5 flex-1 rounded-full", i < bwd ? "bg-pin" : "bg-muted")} />)}</div>
    </div>
  );
}

interface GraphProps {
  g: ReturnType<typeof layout>; grads: Record<string, number>; reached: Set<string>; known: (v: Value) => boolean; gScale: number;
  lastF: Value | null; lastB: Value | null; rootLive: boolean; sel: string | null; onSel: (l: string) => void; lossLabel: string;
  leafKind: (k: string) => string | undefined;
}

function Graph({ g, grads, reached, known, gScale, lastF, lastB, rootLive, sel, onSel, lossLabel, leafKind }: GraphProps) {
  const [ref, W] = useWidth<HTMLDivElement>(800);
  const vertical = W < 560;
  const nL = g.nLayers, maxW = g.maxWidth;
  // node box + spacing
  const nw = vertical ? Math.min(92, Math.floor((W - 28) / maxW) - 8) : Math.min(94, Math.floor((W - 48) / nL) - 22);
  const nh = 60;
  const P = vertical ? 14 : 24; // outer padding (op badges stick out of the boxes)
  const gapMain = vertical ? 30 : Math.max(18, (W - 2 * P - nL * nw) / Math.max(1, nL - 1));
  const gapCross = vertical ? Math.max(6, Math.min(24, (W - maxW * nw) / Math.max(1, maxW))) : 16;
  const mainLen = nL * (vertical ? nh : nw) + (nL - 1) * (vertical ? gapMain : gapMain);
  const crossLen = maxW * (vertical ? nw : nh) + (maxW - 1) * gapCross;
  const SW = vertical ? W : Math.max(W, mainLen + 2 * P), SH = vertical ? mainLen + 2 * P : crossLen + 2 * P;
  const pos = new Map<string, { x: number; y: number }>();
  for (const n of g.nodes) {
    const main = n.layer * ((vertical ? nh : nw) + gapMain);
    const cross = (vertical ? SW : SH) / 2 + n.slot * ((vertical ? nw : nh) + gapCross);
    pos.set(n.v.label, vertical ? { x: cross - nw / 2, y: main + P } : { x: main + (SW - mainLen) / 2, y: cross - nh / 2 });
  }
  const anchor = (l: string, out: boolean) => {
    const p = pos.get(l)!;
    return vertical ? { x: p.x + nw / 2, y: out ? p.y + nh : p.y } : { x: out ? p.x + nw : p.x, y: p.y + nh / 2 };
  };
  const activeF = new Set(lastF ? lastF.prev.map((c) => `${c.label}>${lastF.label}`) : []);
  const activeB = new Set(lastB ? lastB.prev.map((c) => `${c.label}>${lastB.label}`) : []);
  return (
    <div ref={ref} className="w-full overflow-x-auto rounded-xl border bg-[var(--viz-stage)]"
      style={{ ["--grad-pos" as string]: div(1, 1), ["--grad-neg" as string]: div(-1, 1) }}>
      <svg width={SW} height={SH} className="block" role="img" aria-label="computational graph">
        {g.edges.map((e) => {
          const a = anchor(e.from, true), b = anchor(e.to, false), key = `${e.from}>${e.to}`;
          const mid = vertical ? `C${a.x},${(a.y + b.y) / 2} ${b.x},${(a.y + b.y) / 2} ${b.x},${b.y}` : `C${(a.x + b.x) / 2},${a.y} ${(a.x + b.x) / 2},${b.y} ${b.x},${b.y}`;
          const d = `M${a.x},${a.y} ${mid}`;
          const f = activeF.has(key), bk = activeB.has(key);
          const childG = grads[e.from];
          return (
            <g key={key}>
              <path d={d} fill="none" stroke="var(--border)" strokeWidth={1.5} />
              {(f || bk) && <path d={d} fill="none" stroke={f ? "var(--viz-a)" : "var(--viz-pin)"} strokeWidth={3} strokeLinecap="round" className={f ? "bp-fwd" : "bp-bwd"} />}
              {bk && lastB && Math.abs(vertical ? b.y - a.y : b.x - a.x) >= 40 && (() => {
                const [rule, val] = localRule(lastB, lastB.prev.find((c) => c.label === e.from)!);
                const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
                const t = `×${fmt(val)}`, tw = t.length * 6.4 + 8;
                return <g pointerEvents="none"><rect x={mx - tw / 2} y={my - 8} width={tw} height={16} rx={8} fill="var(--viz-pin)" />
                  <text x={mx} y={my + 3.5} fontSize={10} textAnchor="middle" fill="#1a1405" className="font-mono font-semibold">{t}</text>
                  <title>{`local slope ${rule} = ${fmt(val)}; ${e.from} gets ${fmt(childG)}`}</title></g>;
              })()}
            </g>
          );
        })}
        {g.nodes.map(({ v }) => {
          const p = pos.get(v.label)!, isLeaf = !v.prev.length, kind = isLeaf ? leafKind(v.label) : undefined;
          const hasG = reached.has(v.label), gv = grads[v.label], k = known(v);
          const hot = (lastF === v) || (lastB === v) || (rootLive && !g.topo.some((u) => u.prev.includes(v)));
          const fill = hasG ? `color-mix(in srgb, ${div(gv, gScale)} 55%, var(--card))` : "var(--card)";
          const ask = `${isLeaf ? `leaf ${v.label}` : `node ${v.label} = ${OP_NAME[v.op]}(${v.prev.map((c) => c.label).join(", ")})`}: value ${k ? fmt(v.data, 3) : "not computed yet"}, gradient ∂${lossLabel}/∂${v.label} ${hasG ? `= ${fmt(gv, 3)}` : "not computed yet"}`;
          return (
            <g key={v.label} transform={`translate(${p.x},${p.y})`} className="cursor-pointer" onClick={() => onSel(v.label)} data-ask-id={`bp-node-${v.label}`} data-ask={ask}>
              <rect width={nw} height={nh} rx={10} fill={fill} stroke={hot ? (bwdish(lastB, v, rootLive) ? "var(--viz-pin)" : "var(--viz-a)") : sel === v.label ? "var(--foreground)" : "var(--border)"}
                strokeWidth={hot || sel === v.label ? 2.5 : 1} strokeDasharray={kind && kind !== "param" ? "4 3" : undefined} />
              {!isLeaf && <g><circle cx={0} cy={0} r={11} fill="var(--viz-a)" /><text x={0} y={v.op === "tanh" ? 3 : 4} fontSize={v.op === "tanh" ? 7.5 : 12} textAnchor="middle" fill="var(--background)" fontWeight={700}>{OP_NAME[v.op]}</text></g>}
              <text x={nw / 2} y={17} fontSize={12.5} textAnchor="middle" className="fill-foreground font-mono" fontWeight={700}>{v.label}</text>
              <text x={nw / 2} y={34} fontSize={11.5} textAnchor="middle" className={cn("font-mono tabular-nums", k ? "fill-foreground" : "fill-muted-foreground")}>{k ? fmt(v.data) : "?"}</text>
              <text x={nw / 2} y={50} fontSize={10.5} textAnchor="middle" className={cn("font-mono tabular-nums", hasG ? "fill-foreground" : "fill-muted-foreground")} fontWeight={hasG ? 700 : 400}>{hasG ? `∇ ${fmt(gv)}` : "∇ ?"}</text>
            </g>
          );
        })}
      </svg>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t px-3 py-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="size-3 rounded-sm" style={{ background: div(-1, 1) }} />∇ &lt; 0: increasing it lowers {lossLabel}</span>
        <span className="flex items-center gap-1.5"><span className="size-3 rounded-sm" style={{ background: div(1, 1) }} />∇ &gt; 0: increasing it raises {lossLabel}</span>
        <span className="flex items-center gap-1.5"><span className="h-3 w-4 rounded-sm border border-dashed border-muted-foreground" />data (not learned)</span>
      </div>
    </div>
  );
}
const bwdish = (lastB: Value | null, v: Value, rootLive: boolean) => lastB === v || rootLive;

function NodeDetail({ v, preset, vals, live, grad, hasGrad, known, onClose }: { v: Value; preset: Preset; vals: Record<string, number>; live: boolean; grad: number; hasGrad: boolean; known: boolean; onClose: () => void }) {
  const isLeaf = !v.prev.length;
  const eps = 0.1;
  const L0 = preset.build(vals).data;
  const L1 = isLeaf ? preset.build({ ...vals, [v.label]: vals[v.label] + eps }).data : NaN;
  return (
    <div className="rounded-xl border bg-card p-3 text-sm" data-ask-id={`bp-detail-${v.label}`} data-ask={`detail for ${v.label}: value ${fmt(v.data, 3)}, gradient ${fmt(grad, 3)}${isLeaf ? `; nudging by +0.1 predicted change ${fmt(grad * eps, 4)}, actual ${fmt(L1 - L0, 4)}` : ""}`}>
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-base font-semibold">{v.label}</span>
        <Badge variant="secondary">{isLeaf ? preset.leaves.find((l) => l.key === v.label)?.hint : `${OP_NAME[v.op]} node`}</Badge>
        <Button variant="ghost" size="sm" className="ml-auto" onClick={onClose}>Close</Button>
      </div>
      <div className="grid gap-3 @xl:grid-cols-2">
        <div>
          <div className="text-xs text-muted-foreground">Forward</div>
          <div className="font-mono text-[13px]">{isLeaf ? `${v.label} = ${fmt(v.data, 3)} (you set this)` : known ? forwardText(v) : "not computed yet"}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Gradient ∂{preset.lossLabel}/∂{v.label}</div>
          <div className="font-mono text-[13px]">{hasGrad ? fmt(grad, 4) : "not reached yet"}</div>
        </div>
      </div>
      {isLeaf && live && (
        <div className="mt-3 rounded-lg bg-muted/50 p-2.5 text-[13px]">
          <div className="mb-1 font-medium">Check it yourself: nudge {v.label} by +{eps}</div>
          <div className="grid grid-cols-2 gap-2 font-mono tabular-nums">
            <div><div className="text-xs text-muted-foreground font-sans">gradient predicts</div>Δ{preset.lossLabel} ≈ {fmt(grad, 3)} × {eps} = {fmt(grad * eps, 4)}</div>
            <div><div className="text-xs text-muted-foreground font-sans">actually</div>Δ{preset.lossLabel} = {fmt(L1 - L0, 4)}</div>
          </div>
          <div className="mt-1 text-xs text-muted-foreground">Close, not exact: the gradient is the slope right here, and curves bend. Smaller nudges agree better.</div>
        </div>
      )}
    </div>
  );
}

function Sparkline({ values, label }: { values: number[]; label: string }) {
  if (values.length < 2) return <div className="flex h-24 items-center justify-center rounded-lg border border-dashed text-xs text-muted-foreground">{label} history appears here</div>;
  const W = 300, H = 90, lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const X = (i: number) => 4 + (i / (values.length - 1)) * (W - 8), Y = (v: number) => 8 + (1 - (v - lo) / span) * (H - 22);
  const first = values[0], last = values[values.length - 1];
  return (
    <div data-ask-id="bp-loss-history" data-ask={`${label} over ${values.length - 1} gradient steps: from ${fmt(first, 4)} to ${fmt(last, 4)}`}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <line x1={4} x2={W - 4} y1={Y(lo)} y2={Y(lo)} stroke="var(--viz-grid)" />
        <path d={values.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join("")} fill="none" stroke="var(--viz-a)" strokeWidth={2} strokeLinejoin="round" />
        <circle cx={X(values.length - 1)} cy={Y(last)} r={3.5} fill="var(--viz-a)" />
        <text x={4} y={H - 2} fontSize={10} className="fill-muted-foreground">{values.length - 1} steps</text>
      </svg>
      <div className="flex justify-between font-mono text-xs tabular-nums"><span className="text-muted-foreground">{label} {fmt(first, 3)}</span><span>→ {fmt(last, 3)}</span></div>
    </div>
  );
}
