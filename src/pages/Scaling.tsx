import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, BookOpen } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { setScreen } from "../lib/store";
import { isDark } from "../lib/theme";
import { Legend, LineChart, Stat, Waiting, human, logExtent, useResearch, type Series } from "../labs/charts";
import { ComputeCalc } from "../labs/ScalingCalc";

interface Run {
  id: string; d_model: number; n_layer: number; n_head: number; d_mlp?: number; params: number; non_embedding_params: number;
  tokens_seen: number; flops: number; final_val: number | null; final_train?: number | null; seconds?: number;
  curve: { step: number[]; tokens: number[]; flops?: number[]; val: (number | null)[]; train?: (number | null)[] };
}
interface ScalingData { steps: number; batch?: number; tokens_per_step?: number; note?: string; runs: Run[]; fit?: { formula: string; a: number; alpha: number; c: number; fit_on: string; r2?: number; pred?: number[] } }

/** Sequential single-hue ramp: small models faint, big models strong. */
export function sizeColor(i: number, n: number) {
  const t = n <= 1 ? 1 : i / (n - 1);
  return isDark() ? `oklch(${(0.5 + 0.35 * t).toFixed(3)} ${(0.09 + 0.08 * t).toFixed(3)} ${(250 - 30 * t).toFixed(0)})`
    : `oklch(${(0.8 - 0.42 * t).toFixed(3)} ${(0.08 + 0.12 * t).toFixed(3)} ${(240 + 25 * t).toFixed(0)})`;
}
const nOf = (r: Run, key: string) => (typeof (r as unknown as Record<string, unknown>)[key] === "number" ? (r as unknown as Record<string, number>)[key] : r.params);

export function Scaling() {
  const load = useResearch<ScalingData>("/research/scaling.json");
  const d = load.data;
  const [parts, setParts] = useState({ curves: "", law: "", calc: "" });
  const onCurves = useCallback((s: string) => setParts((p) => (p.curves === s ? p : { ...p, curves: s })), []);
  const onLaw = useCallback((s: string) => setParts((p) => (p.law === s ? p : { ...p, law: s })), []);
  const onCalc = useCallback((s: string) => setParts((p) => (p.calc === s ? p : { ...p, calc: s })), []);
  useEffect(() => {
    setScreen("Scaling", d ? "Scaling experiment" : "Calculator", [
      d ? `Real scaling experiment: ${d.runs.length} TinyGPT-family models from ${human(Math.min(...d.runs.map((r) => r.params)))} to ${human(Math.max(...d.runs.map((r) => r.params)))} parameters, each trained ${d.steps} steps on the same ${human(d.runs[0].tokens_seen)} tokens. ${d.note ?? ""}` : "Scaling experiment results (public/research/scaling.json) are not available yet.",
      parts.curves, parts.law, parts.calc,
    ].filter(Boolean).join("\n\n"));
  }, [d, parts]);

  const runs = useMemo(() => (d ? [...d.runs].sort((a, b) => a.params - b.params) : []), [d]);
  return (
    <div className="space-y-6 pb-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Scaling</h1>
        <p className="mt-1 max-w-3xl text-muted-foreground">Make the model bigger and the loss goes down, smoothly enough to fit with a formula. That one observation is why labs build ever larger models. Here it is measured on TinyGPT's family, plus the math to plan your own run.</p>
      </div>
      {!d ? <Waiting what="Scaling runs" file="public/research/scaling.json" load={load} /> : <>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat k="model sizes" v={runs.length} sub="same recipe, same data" />
          <Stat k="parameters" v={`${human(runs[0].params)} → ${human(runs[runs.length - 1].params)}`} sub={`${Math.round(runs[runs.length - 1].params / runs[0].params)}× range`} />
          <Stat k="tokens per run" v={human(runs[0].tokens_seen)} sub={`${d.steps.toLocaleString()} steps`} />
          <Stat k="best val loss" v={Math.min(...runs.map((r) => r.final_val ?? Infinity)).toFixed(2)} sub={`biggest model · chance is 5.55`} />
        </div>
        <LossCurves runs={runs} onSummary={onCurves} />
        <PowerLaw d={d} runs={runs} onSummary={onLaw} />
      </>}
      <ComputeCalc onSummary={onCalc} />
      <Alert>
        <BookOpen />
        <AlertTitle>Where this comes from</AlertTitle>
        <AlertDescription>
          <p>Kaplan et al. 2020, <a className="underline" href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noreferrer">Scaling Laws for Neural Language Models</a>, found loss falls as a power law in parameters, data and compute over many orders of magnitude. Hoffmann et al. 2022, <a className="underline" href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noreferrer">Training Compute-Optimal LLMs</a> ("Chinchilla"), refined how to split a budget between size and data.</p>
        </AlertDescription>
      </Alert>
    </div>
  );
}

function LossCurves({ runs, onSummary }: { runs: Run[]; onSummary: (s: string) => void }) {
  const [xMode, setXMode] = useState<"tokens" | "compute">("tokens");
  const [xLog, setXLog] = useState(true);
  const series: Series[] = runs.map((r, i) => {
    const xs = xMode === "tokens" ? r.curve.tokens : r.curve.flops ?? r.curve.tokens.map((t) => 6 * r.params * t);
    return { id: r.id, label: `${human(r.params)} (${r.id})`, color: sizeColor(i, runs.length), width: 2,
      points: xs.map((x, j) => [x, r.curve.val[j] ?? NaN] as [number, number]).filter(([, y]) => Number.isFinite(y)) };
  });
  const finals = runs.map((r) => r.final_val ?? NaN).filter(Number.isFinite);
  const yDomain: [number, number] = [Math.max(0, Math.min(...finals) * 0.9), Math.min(5.7, Math.max(...finals) * 2 + 1)];
  // which model wins at the smallest compute budget where all runs have data?
  const budget = xMode === "compute" ? Math.min(...runs.map((r) => (r.curve.flops ?? r.curve.tokens.map((t) => 6 * r.params * t)).at(-1)!)) : 0;
  const atBudget = xMode === "compute" ? runs.map((r, i) => {
    const pts = series[i].points.filter(([x]) => x <= budget);
    return { r, v: pts.length ? pts[pts.length - 1][1] : NaN };
  }).filter((x) => Number.isFinite(x.v)).sort((a, b) => a.v - b.v)[0] : null;

  const big = runs.slice(-3).map((r) => r.final_val ?? NaN);
  const tied = big.length === 3 && big.every(Number.isFinite) && Math.max(...big) - Math.min(...big) < 0.03 ? { n: 3, spread: Math.max(...big) - Math.min(...big) } : null;
  useEffect(() => {
    onSummary(`Loss curves (validation loss vs ${xMode === "tokens" ? "tokens seen" : "training compute in FLOPs"}, ${xLog ? "log" : "linear"} x-axis). Final val loss by size: ${runs.map((r) => `${r.id} ${human(r.params)} params → ${r.final_val?.toFixed(3)}`).join("; ")}.` +
      (atBudget ? ` At ${budget.toExponential(1)} FLOPs (the smallest model's whole budget) the best curve is ${atBudget.r.id} (${human(atBudget.r.params)} params) at ${atBudget.v.toFixed(3)}.` : ""));
  }, [xMode, xLog, runs, atBudget, budget, onSummary]);

  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle>Bigger models learn faster from the same data</CardTitle>
        <CardDescription>Validation loss for every size, trained with the identical recipe. Tap or hover the chart to read values.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <ToggleGroup type="single" variant="outline" value={xMode} onValueChange={(v) => v && setXMode(v as typeof xMode)}>
            <ToggleGroupItem value="tokens" className="h-9 px-3">vs tokens</ToggleGroupItem>
            <ToggleGroupItem value="compute" className="h-9 px-3">vs compute</ToggleGroupItem>
          </ToggleGroup>
          <div className="flex items-center gap-2"><Switch id="sc-log" checked={xLog} onCheckedChange={setXLog} /><Label htmlFor="sc-log">log x-axis</Label></div>
        </div>
        <LineChart series={series} xLog={xLog} yDomain={yDomain} xLabel={xMode === "tokens" ? "tokens seen" : "training FLOPs"} yLabel="val loss" height={320} yFmt={(v) => v.toFixed(1)} tipFmt={(v) => v.toFixed(3)}
          xDomain={xLog ? logExtent(series.flatMap((s) => s.points.map((p) => p[0])).filter((x) => x > 0), 0.01) : undefined}
          xFmt={(v) => (xMode === "compute" ? v.toExponential(0).replace("e+", "e") : human(v))} />
        <div data-ask-id="sc-curves" data-ask={`validation loss curves for ${runs.length} model sizes vs ${xMode}: ${runs.map((r) => `${r.id} (${r.params} params) final ${r.final_val?.toFixed(3)}`).join(", ")}`}>
          <Legend items={series.map((s) => ({ label: s.label, color: s.color }))} />
        </div>
        <p className="text-sm text-muted-foreground">
          {xMode === "tokens"
            ? <>Read it vertically: at any amount of data, the bigger model (stronger color) has lower loss. They see the same tokens; they just squeeze more out of each one.{tied && <> But look at the right edge: the {tied.n} biggest models finish within {tied.spread.toFixed(3)} of each other. With only {human(runs[0].tokens_seen)} training tokens each, the training budget, not model size, has become the limit: bigger models need more tokens before their extra capacity pays off.</>}</>
            : atBudget ? <>Same x-axis now means same <i>cost</i>. Big models cost more per token, so their curves shift right. With the budget of the whole smallest run ({budget.toExponential(1).replace("e+", "e")} FLOPs), the best choice here was <b className="text-foreground">{human(atBudget.r.params)} params</b> (loss {atBudget.v.toFixed(2)}): neither the smallest nor necessarily the biggest.</> : null}
        </p>
      </CardContent>
    </Card>
  );
}

function PowerLaw({ d, runs, onSummary }: { d: ScalingData; runs: Run[]; onSummary: (s: string) => void }) {
  const fit = d.fit;
  const key = fit?.fit_on ?? "params";
  const [mult, setMult] = useState(10);
  const [sub, setSub] = useState(false);
  const pts = runs.map((r, i) => ({ r, i, n: nOf(r, key), l: r.final_val ?? NaN })).filter((p) => Number.isFinite(p.l));
  const law = (n: number) => (fit ? fit.c + fit.a * n ** -fit.alpha : NaN);
  const nMax = Math.max(...pts.map((p) => p.n)), nMin = Math.min(...pts.map((p) => p.n));
  const nStar = nMax * mult, lStar = law(nStar), lMaxPred = law(nMax);
  const tf = (l: number) => (sub && fit ? l - fit.c : l);
  const best = pts.find((p) => p.n === nMax)!;
  const xDomain: [number, number] = [nMin / 2, Math.max(nMax * 2, nStar * 1.8)];
  const fitPts: [number, number][] = fit ? Array.from({ length: 80 }, (_, i) => { const n = 10 ** (Math.log10(xDomain[0]) + (i / 79) * Math.log10(xDomain[1] / xDomain[0])); return [n, tf(law(n))]; }) : [];
  const ys = [...pts.map((p) => tf(p.l)), ...(fit ? [tf(lStar)] : [])].filter((y) => y > 0);
  const yDomain = logExtent(ys, 0.25);
  const series: Series[] = fit ? [
    { id: "fit", label: "fit", color: "var(--viz-pin)", points: fitPts.filter(([n]) => n <= nMax * 1.05), width: 2 },
    { id: "extrap", label: "extrapolation", color: "var(--viz-pin)", points: fitPts.filter(([n]) => n >= nMax), dashed: true, width: 2 },
  ] : [];
  const nLabel = key === "non_embedding_params" ? "non-embedding params" : "params";

  useEffect(() => {
    onSummary(fit ? `Power-law fit ${fit.formula} on ${nLabel}: a = ${fit.a.toPrecision(3)}, alpha = ${fit.alpha.toFixed(3)}, c = ${fit.c.toFixed(3)}${fit.r2 != null ? `, R² = ${fit.r2.toFixed(3)}` : ""}. ` +
      `Largest trained model: ${human(nMax)} ${nLabel}, val loss ${best.l.toFixed(3)} (fit says ${lMaxPred.toFixed(3)}). Extrapolation slider at ${mult}× → ${human(nStar)} ${nLabel}: predicted ${lStar.toFixed(3)} if the trend held and data were not limiting.` : "No fit in the data.");
  }, [fit, nLabel, nMax, best, lMaxPred, mult, nStar, lStar, onSummary]);

  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle>A straight line on log–log paper</CardTitle>
        <CardDescription>Final loss vs model size. The fitted curve is {fit ? <span className="font-mono">{fit.formula.replace("alpha", "α")}</span> : "missing"}: a power law plus a floor <span className="font-mono">c</span> no model size can beat on this data.</CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-6 @3xl:grid-cols-5 [&>*]:min-w-0">
        <div className="space-y-2 @3xl:col-span-3">
          <LineChart series={series} xLog yLog xDomain={xDomain} yDomain={yDomain} xLabel={nLabel} yLabel={sub ? "loss − c" : "final val loss"} height={320} crosshair={false}
            yFmt={(v) => (v >= 1 ? v.toFixed(1) : v.toPrecision(2))}
            overlay={(x, y) => <>
              {pts.map((p) => (
                <g key={p.r.id} data-ask-id={`sc-pt-${p.r.id}`} data-ask={`model ${p.r.id}: ${p.n.toLocaleString()} ${nLabel} (${p.r.params.toLocaleString()} total), d_model ${p.r.d_model}, ${p.r.n_layer} layers, final val loss ${p.l.toFixed(3)}${fit ? `, fit predicts ${law(p.n).toFixed(3)}` : ""}`}>
                  <circle cx={x(p.n)} cy={y(tf(p.l))} r={14} fill="transparent" />
                  <circle cx={x(p.n)} cy={y(tf(p.l))} r={5.5} fill={sizeColor(p.i, runs.length)} stroke="var(--background)" strokeWidth={2} />
                  <text x={x(p.n)} y={y(tf(p.l)) + (fit && p.l < law(p.n) ? 17 : -10)} fontSize={10.5} textAnchor="middle" className="fill-muted-foreground">{p.r.id}</text>
                </g>
              ))}
              {fit && tf(lStar) > 0 && (
                <g data-ask-id="sc-extrap" data-ask={`extrapolated: ${human(nStar)} ${nLabel} → predicted loss ${lStar.toFixed(3)}`}>
                  <circle cx={x(nStar)} cy={y(tf(lStar))} r={7} fill="var(--background)" stroke="var(--viz-pin)" strokeWidth={2.5} />
                  <text x={x(nStar)} y={y(tf(lStar)) + 22} fontSize={11} textAnchor="middle" className="fill-foreground font-medium">{mult}×?</text>
                </g>
              )}
            </>} />
          <Legend items={[{ label: "trained models (measured)", color: "var(--viz-a)", dot: true }, { label: "power-law fit", color: "var(--viz-pin)" }, { label: "extrapolation", color: "var(--viz-pin)", dashed: true }]} />
          {fit && <div className="flex items-center gap-2"><Switch id="sc-sub" checked={sub} onCheckedChange={setSub} /><Label htmlFor="sc-sub" className="text-sm">Subtract the floor c = {fit.c.toFixed(2)}: the power law becomes a straight line</Label></div>}
        </div>
        {fit && (
          <div className="space-y-4 @3xl:col-span-2">
            <div className="space-y-2">
              <div className="flex items-baseline justify-between"><Label>What would a bigger model get?</Label><span className="font-mono text-sm tabular-nums">{mult}×</span></div>
              <Slider min={0} max={3} step={0.01} value={[Math.log10(mult)]} onValueChange={([v]) => setMult(+(10 ** v).toPrecision(2))} aria-label="size multiplier" />
              <div className="flex justify-between text-[11px] text-muted-foreground"><span>1×</span><span>10×</span><span>100×</span><span>1000×</span></div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Stat k={`biggest trained (${best.r.id})`} v={best.l.toFixed(3)} sub={`${human(nMax)} ${nLabel}`} />
              <Stat k={`predicted at ${mult}×`} v={lStar.toFixed(3)} sub={`${human(nStar)} ${nLabel}`} className="border-[var(--viz-pin)]/50" />
            </div>
            <div className="rounded-lg bg-muted/40 p-3 text-sm" data-ask-id="sc-fit-params" data-ask={`fit: a=${fit.a}, alpha=${fit.alpha}, c=${fit.c}`}>
              <div className="font-mono text-[13px]">α = {fit.alpha.toFixed(3)} · c = {fit.c.toFixed(3)}{fit.r2 != null && <> · R² = {fit.r2.toFixed(3)}</>}</div>
              <p className="mt-1 text-xs text-muted-foreground">Each 10× in size multiplies the reducible loss (L − c) by 10<sup>−α</sup> = {(10 ** -fit.alpha).toFixed(2)}. The gains shrink as you approach the floor.</p>
            </div>
            <Alert>
              <AlertTriangle />
              <AlertTitle>Be honest about the dashed line</AlertTitle>
              <AlertDescription>
                <ul className="list-disc space-y-1 pl-4 text-xs">
                  <li>It's fit on {pts.length} tiny models spanning {Math.round(nMax / nMin)}×. Extrapolating {mult}× past the data is a guess with growing error bars.</li>
                  <li>Every run saw the same {human(runs[0].tokens_seen)} tokens. A much bigger model would need more data (and compute) to actually reach the prediction.</li>
                  <li>The floor c reflects this toy corpus and recipe; it says nothing about real-world text.</li>
                </ul>
              </AlertDescription>
            </Alert>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
