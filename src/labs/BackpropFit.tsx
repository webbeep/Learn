// "Fit a tiny dataset": a 2-layer MLP trained with the same scalar autograd, live.
import { useCallback, useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { div } from "../lib/viz";
import { DATASETS, Trainer, accuracy, makeData, makeNet, type DatasetId, type Opt } from "./fit";
import { LineChart, Stat, useWidth } from "./charts";

const MAX_STEPS = 3000;

export function FitLab({ onSummary }: { onSummary: (s: string) => void }) {
  const [dsId, setDsId] = useState<DatasetId>("moons");
  const [hidden, setHidden] = useState(8);
  const [opt, setOpt] = useState<Opt>("adam");
  const [lr, setLr] = useState(0.03);
  const [running, setRunning] = useState(false);
  const [losses, setLosses] = useState<number[]>([]);
  const [msPerStep, setMs] = useState(0);
  const tr = useRef<Trainer>(null as unknown as Trainer);
  const [, force] = useState(0);
  const cfg = useRef({ opt, lr }); cfg.current = { opt, lr };

  const rebuild = useCallback((id: DatasetId, h: number) => {
    const ds = makeData(id); tr.current = new Trainer(makeNet(ds, h), ds);
    setLosses([]); setRunning(false); force((x) => x + 1);
  }, []);
  if (!tr.current) { const ds = makeData(dsId); tr.current = new Trainer(makeNet(ds, hidden), ds); }

  const runSteps = useCallback((budgetMs: number, maxN = Infinity) => {
    const t0 = performance.now(), out: number[] = [];
    while (out.length < maxN && (out.length === 0 || performance.now() - t0 < budgetMs)) {
      out.push(tr.current.step(cfg.current.opt, cfg.current.lr));
    }
    setMs((performance.now() - t0) / out.length);
    setLosses((l) => [...l, ...out]);
  }, []);

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const loop = () => { runSteps(14); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, runSteps]);
  useEffect(() => { if (losses.length >= MAX_STEPS) setRunning(false); }, [losses.length]);

  const t = tr.current, ds = t.ds;
  const acc = accuracy(t.net, ds), nParams = t.params.length;
  const cur = losses.length ? losses[losses.length - 1] : NaN;
  const meta = DATASETS.find((d) => d.id === dsId)!;

  // while training, refresh the tutor's summary only every 200 steps (not every frame)
  const summaryTick = running ? Math.floor(losses.length / 200) : losses.length + 1e6;
  useEffect(() => {
    onSummary(`Fit-a-tiny-dataset mode: dataset "${meta.name}" (${meta.blurb}), MLP ${ds.dim}→${hidden}→${hidden}→1 tanh with ${nParams} parameters, optimizer ${opt} lr ${lr}. ` +
      `Steps ${losses.length}, loss ${losses.length ? `${losses[0].toFixed(4)} → ${cur.toFixed(4)}` : "not started"}${ds.dim === 2 ? `, training accuracy ${(acc * 100).toFixed(0)}%` : ""}. ` +
      `Every step builds a graph of about ${(nodesPerStep(ds.dim, hidden, ds.X.length) / 1000).toFixed(0)}k scalar nodes and back-propagates through it.`);
  }, [meta, ds, hidden, nParams, opt, lr, summaryTick, onSummary]); // eslint-disable-line

  const curve = downsample(losses, 300);
  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle>Fit a tiny dataset</CardTitle>
        <CardDescription>The same engine, scaled up: a 2-layer net where every weight is a node like the ones above. Each step builds a graph of ~{(nodesPerStep(ds.dim, hidden, ds.X.length) / 1000).toFixed(0)}k nodes, runs it backward, and nudges all {nParams} weights.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ToggleGroup type="single" variant="outline" value={dsId} onValueChange={(v) => { if (v) { setDsId(v as DatasetId); rebuild(v as DatasetId, hidden); } }} className="grid w-full grid-cols-2 @xl:flex">
          {DATASETS.map((d) => <ToggleGroupItem key={d.id} value={d.id} className="h-9 flex-1">{d.name}</ToggleGroupItem>)}
        </ToggleGroup>
        <div className="grid grid-cols-1 gap-5 @3xl:grid-cols-2 [&>*]:min-w-0">
          <div className="space-y-2">
            <FitPlot t={t} version={losses.length} />
            <p className="text-xs text-muted-foreground">{meta.blurb}. {ds.dim === 2 ? "Background = what the net predicts at every point; dots = training data." : "Dots = training data; line = the net's current guess."}</p>
          </div>
          <div className="space-y-4">
            <div className="flex gap-2">
              <Button className="h-9 flex-1" onClick={() => setRunning(!running)} disabled={losses.length >= MAX_STEPS}>{running ? <><Pause />Pause</> : <><Play />{losses.length ? "Resume" : "Train"}</>}</Button>
              <Button variant="outline" className="h-9" onClick={() => runSteps(0, 1)} disabled={running} title="One step"><StepForward /></Button>
              <Button variant="outline" className="h-9" onClick={() => rebuild(dsId, hidden)} title="New random weights"><RotateCcw /></Button>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <Stat k="step" v={losses.length.toLocaleString()} />
              <Stat k="loss" v={Number.isFinite(cur) ? cur.toFixed(3) : "–"} data-ask-id="fit-loss" data-ask={`current training loss ${cur}`} />
              <Stat k={ds.dim === 2 ? "accuracy" : "params"} v={ds.dim === 2 ? `${Math.round(acc * 100)}%` : nParams} />
            </div>
            {curve.length > 1 ? (
              <LineChart series={[{ id: "loss", label: "loss", color: "var(--viz-a)", points: curve }]} yLog xLabel="step" yLabel="loss" height={170}
                yFmt={(v) => (v >= 0.1 ? v.toFixed(1) : v.toPrecision(1))} xFmt={(v) => String(Math.round(v))}
                askId="fit-loss-curve" ask={`training loss curve over ${losses.length} steps: ${losses[0]?.toFixed(3)} → ${cur.toFixed(4)}`} />
            ) : <div className="flex h-[170px] items-center justify-center rounded-lg border border-dashed text-xs text-muted-foreground">Loss curve appears when you press Train</div>}
            <div className="grid grid-cols-1 gap-4 @lg:grid-cols-2">
              <div className="space-y-2">
                <Label>Hidden neurons per layer</Label>
                <ToggleGroup type="single" variant="outline" value={String(hidden)} onValueChange={(v) => { if (v) { setHidden(+v); rebuild(dsId, +v); } }} className="w-full">
                  {[4, 8, 16].map((h) => <ToggleGroupItem key={h} value={String(h)} className="h-9 flex-1">{h}</ToggleGroupItem>)}
                </ToggleGroup>
              </div>
              <div className="space-y-2">
                <Label>Update rule</Label>
                <ToggleGroup type="single" variant="outline" value={opt} onValueChange={(v) => { if (v) { setOpt(v as Opt); setLr(v === "sgd" ? 0.5 : 0.03); } }} className="w-full">
                  <ToggleGroupItem value="sgd" className="h-9 flex-1">SGD</ToggleGroupItem>
                  <ToggleGroupItem value="adam" className="h-9 flex-1">Adam</ToggleGroupItem>
                </ToggleGroup>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm"><Label>Learning rate</Label><span className="font-mono tabular-nums">{lr.toFixed(3)}</span></div>
              <Slider min={-3} max={0.3} step={0.01} value={[Math.log10(lr)]} onValueChange={([v]) => setLr(+(10 ** v).toPrecision(2))} />
              <p className="text-xs text-muted-foreground">{opt === "adam" ? "Adam: a gradient step that adapts per weight (what TinyGPT used). Try 0.3 to see it thrash." : "Plain SGD: w ← w − lr·∇w, exactly the button above."}{msPerStep > 0 && ` · ${msPerStep.toFixed(1)} ms per step on your device`}</p>
            </div>
          </div>
        </div>
        {hidden === 16 && msPerStep > 15 && <p className="text-xs text-muted-foreground">Slower? Every multiply is its own JavaScript object here. Real frameworks batch millions of them into one matrix multiply on a GPU: same math, different bookkeeping.</p>}
      </CardContent>
    </Card>
  );
}

function nodesPerStep(dim: number, h: number, n: number) {
  const neuron = (nin: number) => 2 * nin + 2; // nin muls + nin adds + bias + act
  return n * (dim + h * neuron(dim) + h * neuron(h) + neuron(h) + 4);
}

function downsample(ls: number[], n: number): [number, number][] {
  if (ls.length <= n) return ls.map((l, i) => [i + 1, l]);
  const out: [number, number][] = [], k = ls.length / n;
  for (let i = 0; i < n; i++) { const j = Math.min(ls.length - 1, Math.round(i * k)); out.push([j + 1, ls[j]]); }
  out.push([ls.length, ls[ls.length - 1]]);
  return out;
}

function FitPlot({ t, version }: { t: Trainer; version: number }) {
  const [ref, W] = useWidth<HTMLDivElement>(360);
  const S = Math.min(W, 420);
  const canvas = useRef<HTMLCanvasElement>(null);
  const ds = t.ds;
  const R = ds.dim === 2 ? Math.max(...ds.X.flat().map(Math.abs)) * 1.1 : 3.3; // half-width of the view
  const toPx = (v: number) => ((v + R) / (2 * R)) * S;
  useEffect(() => {
    if (ds.dim !== 2 || !canvas.current) return;
    const c = canvas.current, N = 56, ctx = c.getContext("2d")!;
    c.width = N; c.height = N;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = -R + ((i + 0.5) / N) * 2 * R, y = R - ((j + 0.5) / N) * 2 * R;
      const s = t.net.eval([x, y]);
      ctx.fillStyle = div(Math.tanh(s / 2), 1.6); ctx.fillRect(i, j, 1, 1);
    }
  }, [t, version, ds, R, S]);
  const yPx = (v: number) => S - toPx(v * (ds.dim === 1 ? 2.2 : 1));
  const fitPath = ds.dim === 1 ? Array.from({ length: 121 }, (_, i) => { const x = -R + (i / 120) * 2 * R; return `${i ? "L" : "M"}${toPx(x).toFixed(1)},${yPx(t.net.eval([x])).toFixed(1)}`; }).join("") : "";
  const truth = ds.dim === 1 ? Array.from({ length: 121 }, (_, i) => { const x = -R + (i / 120) * 2 * R; return `${i ? "L" : "M"}${toPx(x).toFixed(1)},${yPx(Math.sin(1.5 * x)).toFixed(1)}`; }).join("") : "";
  return (
    <div ref={ref} className="flex w-full justify-center">
      <div className="relative overflow-hidden rounded-xl border bg-[var(--viz-stage)]" style={{ width: S, height: S }}
        data-ask-id="fit-plot" data-ask={ds.dim === 2 ? `decision boundary of the MLP on the ${ds.name} dataset; orange = class +1, blue = class −1` : "the MLP's fitted curve versus the noisy sine data"}>
        {ds.dim === 2 && <canvas ref={canvas} className="absolute inset-0 size-full opacity-80" style={{ imageRendering: "auto" }} />}
        <svg width={S} height={S} className="absolute inset-0">
          {ds.dim === 1 && <>
            <line x1={0} x2={S} y1={yPx(0)} y2={yPx(0)} stroke="var(--viz-grid)" />
            <path d={truth} fill="none" stroke="var(--muted-foreground)" strokeOpacity={0.5} strokeDasharray="4 4" strokeWidth={1.5} />
            <path d={fitPath} fill="none" stroke="var(--viz-a)" strokeWidth={2.5} strokeLinejoin="round" />
          </>}
          {ds.X.map((p, i) => ds.dim === 2
            ? <circle key={i} cx={toPx(p[0])} cy={S - toPx(p[1])} r={5} fill={div(ds.Y[i], 1)} stroke="var(--background)" strokeWidth={1.5} />
            : <circle key={i} cx={toPx(p[0])} cy={yPx(ds.Y[i])} r={4.5} fill="var(--viz-pin)" stroke="var(--background)" strokeWidth={1.5} />)}
        </svg>
        {ds.dim === 1 && <div className="absolute right-2 bottom-1.5 flex gap-3 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1"><span className="h-0.5 w-3 bg-viz-a" />net</span>
          <span className="flex items-center gap-1"><span className="w-3 border-t border-dashed border-muted-foreground" />true sin</span></div>}
      </div>
    </div>
  );
}
