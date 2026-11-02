import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { ChevronDown, CircleAlert, Dices, FileText, Pause, Play, RotateCcw, Settings2, StepForward } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useIsMobile } from "@/hooks/use-mobile";
import { cn } from "@/lib/utils";
import { setScreen, useStore } from "../lib/store";
import { pct, seq, show } from "../lib/viz";
import { DEFAULT_CONFIG, DEFAULT_LR, OPTIM_LABEL, PRESETS, PROBE, applyParams, paramCount, parseHashQuery, withPreset, type Optim, type PresetId, type TrainConfig } from "../train/config";
import { ensureStarted, pause, reset, setCfg, setData, start, stepOnce, trainer, type TrainState } from "../train/client";
import { LineChart, useWidth } from "../train/charts";
import type { Viz } from "../train/trainer";

const LN256 = Math.log(256);
const fmtLr = (lr: number) => (lr >= 0.1 ? lr.toFixed(lr >= 1 ? 1 : 2) : lr.toExponential(0).replace("e-", "e-"));
const fmtNum = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e4 ? `${(n / 1e3).toFixed(0)}k` : n.toLocaleString());
const fmtTime = (s: number) => (s < 60 ? `${s.toFixed(0)}s` : `${Math.floor(s / 60)}m ${String(Math.floor(s % 60)).padStart(2, "0")}s`);
const fmtG = (g: number) => (g === 0 ? "0" : g >= 0.01 && g < 100 ? g.toFixed(g < 1 ? 3 : 2) : g.toExponential(1));
function ema(xs: number[], a = 0.12) { const out: number[] = []; let m = xs[0] ?? 0; for (const x of xs) { m = isFinite(x) ? (out.length ? (1 - a) * m + a * x : x) : x; out.push(m); } return out; }
function downsample(xs: number[], max = 500) {
  if (xs.length <= max) return { x: xs.map((_, i) => i + 1), y: xs };
  const k = xs.length / max, x: number[] = [], y: number[] = [];
  for (let i = 0; i < max; i++) { const a = Math.floor(i * k), b = Math.floor((i + 1) * k); let s = 0; for (let j = a; j < b; j++) s += xs[j]; x.push(b); y.push(s / (b - a)); }
  return { x, y };
}
const partsText = (c: TrainConfig) => {
  const off = (["attention", "mlp", "pos", "residual", "layernorm"] as const).filter((k) => !c[k]).map((k) => ({ attention: "attention", mlp: "MLP", pos: "positions", residual: "residual", layernorm: "LayerNorm" })[k]);
  return off.length ? `without ${off.join(", ")}` : "all parts on";
};

export function TrainLab() {
  const s = useStore(trainer, (x) => x);
  // load ?config from the hash (e.g. links from the Architecture lab), else start the default run
  useEffect(() => {
    const apply = () => {
      const q = parseHashQuery(location.hash);
      const key = q ? JSON.stringify(q) : "";
      if (q && key !== trainer.get().appliedQuery) {
        trainer.set({ appliedQuery: key });
        reset(applyParams(DEFAULT_CONFIG, q));
        toast("Loaded that setup", { description: "Press Start to train it from scratch." });
      } else ensureStarted();
    };
    apply();
    window.addEventListener("hashchange", apply);
    return () => { window.removeEventListener("hashchange", apply); pause(); };
  }, []);

  const smooth = useMemo(() => ema(s.train), [s.train]);
  const lossNow = smooth.length ? smooth[smooth.length - 1] : NaN;
  const valNow = s.val.length ? s.val[s.val.length - 1].loss : NaN;

  useEffect(() => {
    const c = s.cfg;
    setScreen("Train it yourself", `${s.status === "running" ? "Training" : s.status} · step ${Math.floor(s.step / 10) * 10}`, [
      `The learner is training a GPT from random init in their browser (a TypeScript autograd engine in a Web Worker). Status: ${s.status}${s.status === "diverged" ? " (loss exploded; needs a lower learning rate and a reset)" : ""}.`,
      `Model: ${c.nLayer} layer(s), d_model ${c.dModel}, ${c.nHead} heads, MLP ${4 * c.dModel}, context ${c.nCtx}, vocab 256 (same BPE tokenizer as the real TinyGPT), ${s.params.toLocaleString()} params, ${partsText(c)}.`,
      `Training: ${OPTIM_LABEL[c.optimizer]}, lr ${c.lr} (30-step warmup), batch ${c.batch}×${c.nCtx} tokens, grad clip ${c.clip ? "1.0" : "off"}, seed ${c.seed}. Data: ${s.dataKind === "tiny" ? "tiny-world stories" : "the learner's own text"} (${s.tokens.train.toLocaleString()} train / ${s.tokens.val.toLocaleString()} held-out tokens).`,
      `Step ${s.step}, ${s.sps.toFixed(1)} steps/s, ${s.tokensSeen.toLocaleString()} tokens seen, ${s.elapsed.toFixed(0)} s. Train loss (smoothed) ${isFinite(lossNow) ? lossNow.toFixed(3) : "n/a"}, validation loss ${isFinite(valNow) ? valNow.toFixed(3) : "n/a"} (chance = ln 256 = 5.55; real TinyGPT reached 0.46 after 3000 steps of batch 64).`,
      s.viz ? `Probe "${s.probe || PROBE}" → top guesses ${s.viz.probe.map((p) => `"${p.token}" ${pct(p.p)}`).join(", ")}${s.dataKind === "tiny" ? ` (real TinyGPT: " red" 87%)` : ""}.` : "",
      s.viz ? `Gradient norms: ${s.viz.groups.map((g) => `${g.name} ${fmtG(g.grad)}`).join(", ")}.` : "",
      s.samples ? `Latest samples (step ${s.samples.step}): ${s.samples.list.map((x) => `"${x.prompt}" → "${x.text}"`).join(" | ")}` : "",
    ].filter(Boolean).join("\n"));
  }, [s.status, Math.floor(s.step / 10), s.cfg, s.params, s.samples, s.dataKind]); // eslint-disable-line

  return (
    <div className="@container space-y-5 pb-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Train a GPT yourself</h1>
        <p className="mt-1 max-w-3xl text-muted-foreground">A brand-new model, filled with random numbers, learning to write in your browser. Same tokenizer and stories as the real TinyGPT. Press Start, then break it on purpose.</p>
      </div>
      <ControlBar s={s} lossNow={lossNow} valNow={valNow} />
      {s.status === "error" && (
        <Alert variant="destructive"><CircleAlert /><AlertTitle>Training stopped</AlertTitle><AlertDescription>{s.error}</AlertDescription></Alert>
      )}
      {s.status === "diverged" && (
        <Alert className="border-destructive/50" data-ask-id="train-diverged" data-ask={`The run diverged at step ${s.step}: the loss exploded to ${s.train[s.train.length - 1]?.toFixed(1)} with learning rate ${s.cfg.lr}.`}>
          <CircleAlert className="text-destructive!" /><AlertTitle>It diverged at step {s.step}</AlertTitle>
          <AlertDescription>
            <p>The learning rate ({fmtLr(s.cfg.lr)}) made the updates so big that the loss exploded. Once the weights blow up there is no way back.</p>
            <Button size="sm" className="mt-2" onClick={() => reset({ ...s.cfg, lr: DEFAULT_LR[s.cfg.optimizer] })}><RotateCcw />Reset with lr {fmtLr(DEFAULT_LR[s.cfg.optimizer])}</Button>
          </AlertDescription>
        </Alert>
      )}
      <div className="grid gap-4 @5xl:grid-cols-[300px_minmax(0,1fr)]">
        <Settings s={s} />
        <div className="@container/viz min-w-0 space-y-4">
          <LossCard s={s} smooth={smooth} />
          <div className="grid gap-4 @3xl/viz:grid-cols-2">
            <SamplesCard s={s} />
            <AttnCard s={s} />
            <GradCard s={s} />
            <WeightsCard viz={s.viz} />
          </div>
          <TryCard s={s} />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- control bar
function ControlBar({ s, lossNow, valNow }: { s: TrainState; lossNow: number; valNow: number }) {
  const running = s.status === "running", busy = s.status === "loading";
  const stats: [string, string, string][] = [
    ["Step", s.step.toLocaleString(), `training step ${s.step}`],
    ["Loss", isFinite(lossNow) ? lossNow.toFixed(3) : "—", `smoothed training loss ${lossNow.toFixed(3)} (chance is 5.55)`],
    ["Val loss", isFinite(valNow) ? valNow.toFixed(3) : "—", `validation loss on held-out text: ${valNow.toFixed(3)}`],
    ["Steps/s", running || s.sps ? s.sps.toFixed(1) : "—", `${s.sps.toFixed(1)} training steps per second in this browser`],
    ["Tokens", fmtNum(s.tokensSeen), `${s.tokensSeen.toLocaleString()} training tokens seen so far`],
    ["Time", fmtTime(s.elapsed), `${s.elapsed.toFixed(1)} seconds of training`],
  ];
  return (
    <div className="sticky top-[3.75rem] z-20 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border bg-card/90 p-2 shadow-sm backdrop-blur supports-[backdrop-filter]:bg-card/75">
      <div className="flex items-center gap-1.5">
        <Button id="train-run" className="w-[6.5rem]" onClick={running ? pause : start} disabled={s.status === "diverged" || s.status === "error"}>
          {busy ? <Spinner /> : running ? <Pause className="fill-current" /> : <Play className="fill-current" />}{running ? "Pause" : s.step ? "Resume" : "Start"}
        </Button>
        <Button variant="outline" size="icon" onClick={stepOnce} disabled={running || busy || s.status === "diverged"} title="One training step" aria-label="One training step"><StepForward /></Button>
        <Button variant="outline" size="icon" onClick={() => reset()} title="Reset to random weights (step 0)" aria-label="Reset"><RotateCcw /></Button>
      </div>
      <div className="grid min-w-0 flex-1 grid-cols-3 gap-x-3 gap-y-1 @xl:grid-cols-6">
        {stats.map(([k, v, a]) => (
          <div key={k} className="min-w-0" data-ask-id={`train-stat-${k}`} data-ask={a}>
            <div className="text-[11px] leading-tight text-muted-foreground">{k}</div>
            <div className="truncate text-sm font-semibold tabular-nums">{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- settings
function Settings({ s }: { s: TrainState }) {
  const isMobile = useIsMobile();
  const [open, setOpen] = useState<boolean | null>(null);
  const c = s.cfg, isOpen = open ?? !isMobile;
  useEffect(() => { const f = () => setOpen(true); window.addEventListener("trainlab-open-setup", f); return () => window.removeEventListener("trainlab-open-setup", f); }, []);
  return (
    <Collapsible open={isOpen} onOpenChange={setOpen} asChild>
      <Card className="gap-0 self-start py-0 @5xl:sticky @5xl:top-[8.5rem] @5xl:max-h-[calc(100svh-9.5rem)] @5xl:overflow-y-auto">
        <CollapsibleTrigger asChild>
          <button className="flex w-full items-center gap-2 px-4 py-3 text-left">
            <Settings2 className="size-4 text-muted-foreground" />
            <span className="flex-1"><span className="font-medium">Setup</span>
              <span className="block truncate text-xs text-muted-foreground">{s.params ? `${s.params.toLocaleString()} params` : `~${paramCount(c).toLocaleString()} params`} · {OPTIM_LABEL[c.optimizer]} lr {fmtLr(c.lr)} · {partsText(c)}</span></span>
            <ChevronDown className={cn("size-4 text-muted-foreground transition-transform", isOpen && "rotate-180")} />
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-5 border-t px-4 pt-4 pb-5">
            <Section title="Model size">
              <ToggleGroup type="single" variant="outline" className="w-full" value={c.preset} onValueChange={(v) => v && setCfg(withPreset(c, v as PresetId))}>
                {(Object.keys(PRESETS) as PresetId[]).map((id) => <ToggleGroupItem key={id} value={id} className="h-9 flex-1">{PRESETS[id].label}</ToggleGroupItem>)}
              </ToggleGroup>
              <p className="text-xs text-muted-foreground" data-ask-id="train-size" data-ask={`model: ${c.nLayer} layers, d_model ${c.dModel}, ${c.nHead} heads, context ${c.nCtx}, ${paramCount(c).toLocaleString()} parameters`}>
                {c.nLayer} layer{c.nLayer > 1 ? "s" : ""} · d {c.dModel} · {c.nHead} heads · {paramCount(c).toLocaleString()} params{c.preset === "medium" ? " · slow (a few steps/s)" : ""}
              </p>
              <CommitSlider label="Layers" value={c.nLayer} min={1} max={4} step={1} fmt={(v) => String(v)} onCommit={(v) => setCfg({ nLayer: v, preset: v === PRESETS[c.preset as PresetId]?.nLayer ? c.preset : "custom" })} />
              <CommitSlider label="Context length" value={c.nCtx} min={8} max={64} step={8} fmt={(v) => `${v} tokens`} onCommit={(v) => setCfg({ nCtx: v })} />
            </Section>
            <Section title="Parts of the model">
              {([
                ["attention", "Attention", "tokens look at earlier tokens"],
                ["mlp", "MLP", "per-token processing & memory"],
                ["pos", "Positions", "learned position embeddings"],
                ["residual", "Residual", "x + f(x) instead of f(x)"],
                ["layernorm", "LayerNorm", "re-scale before each part"],
              ] as const).map(([k, label, hint]) => (
                <label key={k} className="flex min-h-9 cursor-pointer items-center gap-3">
                  <span className="flex-1 text-sm leading-tight">{label}<span className="block text-xs text-muted-foreground">{hint}</span></span>
                  <Switch checked={c[k]} onCheckedChange={(v) => setCfg({ [k]: v })} />
                </label>
              ))}
            </Section>
            <Section title="Optimizer">
              <ToggleGroup type="single" variant="outline" className="w-full" value={c.optimizer} onValueChange={(v) => {
                if (!v) return; const o = v as Optim; setCfg({ optimizer: o, lr: DEFAULT_LR[o] });
                toast(`${OPTIM_LABEL[o]}: learning rate set to ${fmtLr(DEFAULT_LR[o])}`, { description: "A sensible start for this optimizer. Applies right away; no restart." });
              }}>
                <ToggleGroupItem value="sgd" className="h-9 flex-1">SGD</ToggleGroupItem>
                <ToggleGroupItem value="momentum" className="h-9 flex-1">Momentum</ToggleGroupItem>
                <ToggleGroupItem value="adamw" className="h-9 flex-1">AdamW</ToggleGroupItem>
              </ToggleGroup>
              <LrSlider lr={c.lr} />
              <div className="space-y-2">
                <div className="flex items-baseline justify-between text-sm"><span>Batch size</span><span className="tabular-nums text-muted-foreground">{c.batch} × {c.nCtx} tokens</span></div>
                <Slider min={0} max={5} step={1} value={[Math.round(Math.log2(c.batch))]} onValueChange={([v]) => setCfg({ batch: 2 ** v })} />
              </div>
              <label className="flex min-h-9 cursor-pointer items-center gap-3">
                <span className="flex-1 text-sm leading-tight">Gradient clipping<span className="block text-xs text-muted-foreground">cap the total gradient length at 1</span></span>
                <Switch checked={c.clip} onCheckedChange={(v) => setCfg({ clip: v })} />
              </label>
              <div className="flex items-center gap-2">
                <Label htmlFor="train-seed" className="flex-1 font-normal">Seed <span className="text-xs text-muted-foreground">(the random start)</span></Label>
                <Input id="train-seed" type="number" min={1} className="h-9 w-20 tabular-nums" value={c.seed} onChange={(e) => { const v = Math.max(1, Math.floor(+e.target.value || 1)); setCfg({ seed: v }); }} />
                <Button variant="outline" size="icon" aria-label="Random seed" title="Random seed" onClick={() => setCfg({ seed: 1 + Math.floor(Math.random() * 9999) })}><Dices /></Button>
              </div>
            </Section>
            <DataSection s={s} />
            <p className="text-xs text-muted-foreground">Size, parts, context and seed rebuild the model from step 0. Learning rate, batch, optimizer and clipping change live.</p>
          </div>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return <div className="space-y-3"><div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{title}</div>{children}</div>;
}
function CommitSlider({ label, value, min, max, step, fmt, onCommit }: { label: string; value: number; min: number; max: number; step: number; fmt: (v: number) => string; onCommit: (v: number) => void }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between text-sm"><span>{label}</span><span className="tabular-nums text-muted-foreground">{fmt(v)}</span></div>
      <Slider min={min} max={max} step={step} value={[v]} onValueChange={([x]) => setV(x)} onValueCommit={([x]) => x !== value && onCommit(x)} />
    </div>
  );
}
function LrSlider({ lr }: { lr: number }) {
  const lo = -4, hi = 0.5, marks = [1e-4, 1e-3, 1e-2, 1e-1, 1];
  return (
    <div className="space-y-2" data-ask-id="train-lr" data-ask={`learning rate ${lr} (log slider from 0.0001 to 3); the real TinyGPT used 0.003 with AdamW`}>
      <div className="flex items-baseline justify-between text-sm"><span>Learning rate</span><span className="font-medium tabular-nums">{fmtLr(lr)}</span></div>
      <Slider min={lo} max={hi} step={0.05} value={[Math.log10(lr)]} onValueChange={([v]) => setCfg({ lr: +(10 ** v).toPrecision(2) })} />
      <div className="relative h-3 text-[10px] text-muted-foreground tabular-nums">
        {marks.map((m) => <span key={m} className="absolute -translate-x-1/2" style={{ left: `${((Math.log10(m) - lo) / (hi - lo)) * 100}%` }}>{m >= 1 ? "1" : m.toExponential(0)}</span>)}
      </div>
    </div>
  );
}
function DataSection({ s }: { s: TrainState }) {
  const [text, setText] = useState(s.customText);
  const norm = text.toLowerCase().replace(/[^\na-z .,?!:']/g, " ");
  const enough = norm.trim().length >= 3000;
  return (
    <Section title="Training text">
      <ToggleGroup type="single" variant="outline" className="w-full" value={s.dataKind} onValueChange={(v) => { if (v === "tiny") setData("tiny"); else if (v === "custom" && s.customText) setData("custom"); else if (v === "custom") toast("Paste some text below, then press Train on this"); }}>
        <ToggleGroupItem value="tiny" className="h-9 flex-1">Tiny world</ToggleGroupItem>
        <ToggleGroupItem value="custom" className="h-9 flex-1">Your text</ToggleGroupItem>
      </ToggleGroup>
      <p className="text-xs text-muted-foreground" data-ask-id="train-data" data-ask={`training text: ${s.dataKind === "tiny" ? "tiny-world sample generated by train/corpus.py" : "the learner's own text"}, ${s.chars.toLocaleString()} characters, ${s.tokens.train.toLocaleString()} training tokens and ${s.tokens.val.toLocaleString()} held-out validation tokens (last 10%)`}>
        {s.dataKind === "tiny" ? <>Synthetic stories from <code>train/corpus.py</code>, {(s.chars / 1000).toFixed(0)} KB · </> : <>Your text · </>}
        {s.tokens.train.toLocaleString()} train + {s.tokens.val.toLocaleString()} held-out tokens. <a className="underline underline-offset-2" href="/train-data/tiny-world.txt" target="_blank" rel="noreferrer"><FileText className="mb-0.5 inline size-3" /> view</a>
      </p>
      <Textarea id="train-text" value={text} onChange={(e) => setText(e.target.value)} rows={4} placeholder="Paste a few pages of any text (a story, song lyrics, code…). It is lowercased and squeezed into the tokenizer's alphabet: a–z, space, newline and . , ? ! : '" className="max-h-48 text-xs" />
      <div className="flex items-center gap-2">
        <span className="flex-1 text-xs text-muted-foreground tabular-nums">{norm.trim().length.toLocaleString()} chars{text && !enough ? " · need 3,000+" : ""}</span>
        <Button size="sm" variant="secondary" disabled={!enough} onClick={() => { setData("custom", norm); toast("Training on your text", { description: "Same 256-token vocabulary, so unusual words get split into small pieces." }); }}>Train on this</Button>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------- loss
function LossCard({ s, smooth }: { s: TrainState; smooth: number[] }) {
  const raw = downsample(s.train), sm = downsample(smooth);
  const all = [...sm.y, ...s.val.map((v) => v.loss)].filter(isFinite);
  const top = Math.max(6, ...all.slice(-200)), yMax = top > 6 ? Math.min(20, Math.ceil(top)) : 6;
  const xMax = Math.max(100, Math.ceil((s.step * 1.15) / 50) * 50);
  const valNow = s.val[s.val.length - 1];
  const ask = `Live loss curve: step ${s.step}; smoothed training loss ${smooth.length ? smooth[smooth.length - 1].toFixed(3) : "n/a"}; validation loss ${valNow ? `${valNow.loss.toFixed(3)} at step ${valNow.step}` : "n/a"}; validation history ${s.val.slice(-8).map((v) => `${v.step}:${v.loss.toFixed(2)}`).join(", ")}. Chance level ln 256 = 5.55, real TinyGPT final 0.46.`;
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>Loss, live</CardTitle>
        <CardDescription>How surprised the model is by the real next token. Validation is measured every 25 steps on text it never trains on.</CardDescription>
      </CardHeader>
      <CardContent>
        <LineChart askId="train-loss" ask={ask} xMax={xMax} yMax={yMax} height={250}
          refs={[{ y: LN256, label: "random guessing (ln 256)" }, { y: 0.46, label: "real TinyGPT, 3,000 steps" }]}
          series={[
            { id: "raw", label: "train (per step)", color: "var(--viz-a)", x: raw.x, y: raw.y, width: 1, opacity: 0.3, hideInTip: true },
            { id: "train", label: "train", color: "var(--viz-a)", x: sm.x, y: sm.y, width: 2.2 },
            { id: "val", label: "validation", color: "var(--viz-b)", x: s.val.map((v) => v.step), y: s.val.map((v) => v.loss), dashed: true, dots: s.val.length < 40 },
          ]} />
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-viz-a" />training loss (smoothed; faint = every step)</span>
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 border-t-2 border-dashed border-viz-b" />validation loss</span>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------- samples
function SamplesCard({ s }: { s: TrainState }) {
  const [pick, setPick] = useState("now");
  const hist = s.history.filter((h) => h.step !== s.samples?.step);
  const cur = pick === "now" ? s.samples : hist.find((h) => String(h.step) === pick) ? { step: +pick, list: hist.find((h) => String(h.step) === pick)!.samples } : s.samples;
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>What it writes {cur ? <>at step {cur.step}</> : "now"}</CardTitle>
        <CardDescription>Fresh samples every 50 steps, same prompts as the real model</CardDescription>
        {s.status === "running" && pick === "now" && <CardAction><Badge variant="secondary" className="gap-1.5"><span className="size-1.5 animate-pulse rounded-full bg-viz-b" />live</Badge></CardAction>}
      </CardHeader>
      <CardContent className="space-y-2">
        {hist.length > 0 && (
          <ToggleGroup type="single" size="sm" variant="outline" className="w-full flex-wrap justify-start" value={pick} onValueChange={(v) => v && setPick(v)}>
            <ToggleGroupItem value="now" className="h-8 px-2.5">latest</ToggleGroupItem>
            {hist.map((h) => <ToggleGroupItem key={h.step} value={String(h.step)} className="h-8 px-2.5 tabular-nums">step {h.step}</ToggleGroupItem>)}
          </ToggleGroup>
        )}
        {cur ? cur.list.map((x, i) => (
          <div key={i} className="rounded-lg border bg-muted/40 p-2.5 font-mono text-[13px] leading-relaxed break-words whitespace-pre-wrap"
            data-ask-id={`train-sample-${i}`} data-ask={`At step ${cur.step} (val loss ${s.val.find((v) => v.step === cur.step)?.loss.toFixed(2) ?? "?"}), given "${x.prompt}" the live model wrote "${x.text}" (temperature 0.8).`}>
            <span className="text-muted-foreground">{x.prompt}</span>{x.text}
          </div>
        )) : <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground"><Spinner />Building the model…</div>}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------- gradients
function GradCard({ s }: { s: TrainState }) {
  const v = s.viz;
  const groups = v ? [...v.groups].reverse() : []; // backward order: output side first
  const lo = -4, hi = 1, frac = (g: number) => Math.max(0.015, Math.min(1, (Math.log10(Math.max(g, 1e-6)) - lo) / (hi - lo)));
  const stream = v ? [...v.stream].reverse() : [];
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>Gradients flowing back</CardTitle>
        <CardDescription>How hard the last step pushed on each part (length of its gradient, log scale). The loss sits at the top; the signal travels down.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!v ? <div className="h-40 animate-pulse rounded-lg bg-muted/50" /> : <>
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">Per part: ‖∂loss/∂weights‖ at step {v.step}</div>
            {groups.map((g) => (
              <div key={g.name} className="grid grid-cols-[5.5rem_1fr_3.5rem] items-center gap-2 text-xs" data-ask-id={`train-grad-${g.name}`}
                data-ask={`Gradient norm of the "${g.name}" parameters at step ${v.step}: ${g.grad.toExponential(2)}. Their weight norm is ${g.weight.toFixed(2)}; between the last two snapshots the weights changed by ${(g.update * 100).toFixed(2)}% of their size.`}>
                <span className="truncate">{g.name}</span>
                <div className="h-3.5 rounded-sm bg-muted"><div className="h-full rounded-sm bg-viz-a transition-[width] duration-300" style={{ width: `${frac(g.grad) * 100}%` }} /></div>
                <span className="text-right tabular-nums">{fmtG(g.grad)}</span>
              </div>
            ))}
            <div className="grid grid-cols-[5.5rem_1fr_3.5rem] gap-2 text-[10px] text-muted-foreground tabular-nums"><span /><span className="flex justify-between"><span>0.0001</span><span>0.01</span><span>1</span><span>10</span></span><span /></div>
          </div>
          <Separator />
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">Along the residual stream: ‖∂loss/∂x‖ arriving at each depth</div>
            {v.step === 0 && <div className="text-xs text-muted-foreground italic">Press Start or ⏵ step once: gradients exist only after a backward pass.</div>}
            {stream.map((g, i) => (
              <div key={g.name} className="grid grid-cols-[5.5rem_1fr_3.5rem] items-center gap-2 text-xs" data-ask-id={`train-stream-${g.name}`}
                data-ask={`Gradient reaching the residual stream ${g.name === "embed" ? "right after the embeddings" : g.name}: norm ${g.grad.toExponential(2)} (summed over the batch). ${s.cfg.residual ? "Residual connections let it pass straight through." : "Residual connections are OFF, so each layer must pass it on through its own weights."}`}>
                <span className="truncate">{i === 0 ? "↓ " : ""}{g.name}</span>
                <div className="h-3.5 rounded-sm bg-muted"><div className="h-full rounded-sm bg-viz-b transition-[width] duration-300" style={{ width: `${frac(g.grad) * 100}%` }} /></div>
                <span className="text-right tabular-nums">{fmtG(g.grad)}</span>
              </div>
            ))}
          </div>
        </>}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------- attention
function AttnCard({ s }: { s: TrainState }) {
  const v = s.viz, [layer, setLayer] = useState(0), [head, setHead] = useState(0);
  const L = s.cfg.nLayer, H = s.cfg.nHead, l = Math.min(layer, L - 1), h = Math.min(head, H - 1);
  const A = v?.attn.layers[l]?.[h];
  const toks = v?.attn.tokens ?? [];
  const [ref, W] = useWidth<HTMLDivElement>(320);
  const T = toks.length, lab = 64, cell = Math.max(6, Math.min(14, Math.floor((W - lab) / Math.max(1, T))));
  const redP = s.pRed, own = s.dataKind === "custom";
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>Attention on a test sentence</CardTitle>
        <CardDescription>Row = the word doing the looking, column = the earlier word it looks at. {own ? "The sentence comes from the held-out part of your text." : "The real model learns to look from the last “the” back to “red”."}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {!s.cfg.attention ? (
          <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">Attention is switched off: every token only sees itself, so the model can't copy {own ? "anything" : "“red”"} from earlier in the sentence.</div>
        ) : (
          <>
            <div className="flex flex-wrap gap-2">
              {L > 1 && <ToggleGroup type="single" size="sm" variant="outline" value={String(l)} onValueChange={(x) => x && setLayer(+x)}>
                {Array.from({ length: L }, (_, i) => <ToggleGroupItem key={i} value={String(i)} className="h-8 px-2.5">L{i + 1}</ToggleGroupItem>)}
              </ToggleGroup>}
              {H > 1 && <ToggleGroup type="single" size="sm" variant="outline" value={String(h)} onValueChange={(x) => x && setHead(+x)}>
                {Array.from({ length: H }, (_, i) => <ToggleGroupItem key={i} value={String(i)} className="h-8 px-2.5">head {i + 1}</ToggleGroupItem>)}
              </ToggleGroup>}
            </div>
            <div ref={ref} className="w-full overflow-hidden">
              {A ? (
                <svg width={lab + cell * T + 2} height={cell * T + 2} className="block">
                  {toks.map((t, i) => (
                    <text key={i} x={lab - 4} y={i * cell + cell / 2 + 3.5} fontSize={Math.min(11, cell - 1)} textAnchor="end" className={cn("fill-muted-foreground font-mono", i === T - 1 && "fill-foreground font-semibold")}>{show(t).slice(0, 8)}</text>
                  ))}
                  {A.map((row, i) => row.map((p, j) => j <= i && (
                    <rect key={`${i}-${j}`} x={lab + j * cell} y={i * cell} width={cell - 1} height={cell - 1} rx={1.5} fill={seq(p)}
                      data-ask-id={`train-attn-${l}-${h}-${i}-${j}`} data-ask={`Layer ${l + 1} head ${h + 1} at step ${v!.step}: token "${toks[i]}" (position ${i}) puts ${pct(p)} of its attention on "${toks[j]}" (position ${j}).`} />
                  )))}
                  {T > 0 && <rect x={lab - 0.5} y={(T - 1) * cell - 0.5} width={cell * T} height={cell} fill="none" stroke="var(--viz-pin)" strokeWidth={1} rx={2} pointerEvents="none" />}
                </svg>
              ) : <div className="h-40 animate-pulse rounded-lg bg-muted/50" />}
            </div>
          </>
        )}
        {v && (
          <div className="@container"><div className={cn("grid gap-3", !own && "@md:grid-cols-2")}>
            <div className="space-y-1" data-ask-id="train-probe" data-ask={`After "${s.probe || PROBE}" the live model (step ${v.step}) predicts: ${v.probe.map((p) => `"${p.token}" ${pct(p.p)}`).join(", ")}.${own ? "" : ` The right answer is " red"; the real TinyGPT gives it 87%.`}`}>
              <div className="truncate text-xs text-muted-foreground">Next token after “…{(s.probe || PROBE).split(" ").slice(-4).join(" ")}”</div>
              {v.probe.map((p) => (
                <div key={p.token} className="grid grid-cols-[4.5rem_1fr_2.6rem] items-center gap-2 text-xs">
                  <span className={cn("truncate font-mono", !own && p.token === " red" && "font-semibold text-pin")}>{show(p.token)}</span>
                  <div className="h-3 rounded-sm bg-muted"><div className={cn("h-full rounded-sm", !own && p.token === " red" ? "bg-pin" : "bg-viz-a")} style={{ width: `${p.p * 100}%` }} /></div>
                  <span className="text-right tabular-nums">{pct(p.p)}</span>
                </div>
              ))}
            </div>
            {!own && <div className="min-w-0">
              <div className="text-xs text-muted-foreground">Chance of “red” over training</div>
              <LineChart askId="train-pred" ask={`Probability of the correct next token " red" on the probe sentence over training; now ${pct(v.pRed)} at step ${v.step} (real TinyGPT: 87%).`}
                height={110} xMax={Math.max(50, redP[redP.length - 1]?.step ?? 50)} yMax={1} fmtY={(x) => pct(x)}
                refs={[{ y: 0.871, label: "real model 87%" }]}
                series={[{ id: "red", label: "p(red)", color: "var(--viz-pin)", x: redP.map((p) => p.step), y: redP.map((p) => p.p) }]} />
            </div>}
          </div></div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------- weights
function WeightsCard({ viz }: { viz: Viz | null }) {
  const [ref, W] = useWidth<HTMLDivElement>(320);
  if (!viz) return <Card><CardHeader><CardTitle>Weights</CardTitle></CardHeader><CardContent><div className="h-40 animate-pulse rounded-lg bg-muted/50" /></CardContent></Card>;
  const hst = viz.hist, n = hst.now.length, Hh = 130, mx = Math.max(...hst.now, ...hst.init), bw = (W - 8) / n;
  const Y = (f: number) => Hh - (Math.sqrt(f / mx) * (Hh - 6));
  const initPath = hst.init.map((f, i) => `${i ? "L" : "M"}${(4 + i * bw).toFixed(1)},${Y(f).toFixed(1)}H${(4 + (i + 1) * bw).toFixed(1)}`).join("");
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>Weights</CardTitle>
        <CardDescription>All weight-matrix numbers inside the blocks, now vs. the random start</CardDescription>
      </CardHeader>
      <CardContent>
        <div ref={ref} data-ask-id="train-weights" data-ask={`Histogram of the block weight matrices at step ${viz.step}: standard deviation ${hst.std.toFixed(4)} now versus ${hst.initStd.toFixed(4)} at random init (normal, std 0.02). Range shown ${hst.lo.toFixed(3)} to ${hst.hi.toFixed(3)}; values beyond are piled into the edge bins.`}>
          <svg width={W} height={Hh + 18} className="block">
            {hst.now.map((f, i) => <rect key={i} x={4 + i * bw + 1} y={Y(f)} width={Math.max(1, bw - 2)} height={Hh - Y(f)} rx={1.5} fill="var(--viz-a)" opacity={0.85} />)}
            <path d={initPath} fill="none" stroke="var(--viz-pin)" strokeWidth={1.5} />
            <line x1={W / 2} x2={W / 2} y1={0} y2={Hh} stroke="var(--muted-foreground)" strokeDasharray="2 3" opacity={0.5} />
            <text x={4} y={Hh + 13} fontSize={10} className="fill-muted-foreground tabular-nums">{hst.lo.toFixed(2)}</text>
            <text x={W / 2} y={Hh + 13} fontSize={10} textAnchor="middle" className="fill-muted-foreground">0</text>
            <text x={W - 4} y={Hh + 13} fontSize={10} textAnchor="end" className="fill-muted-foreground tabular-nums">+{hst.hi.toFixed(2)}</text>
          </svg>
        </div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-viz-a" />now, spread {hst.std.toFixed(3)}</span>
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-pin" />random start, spread {hst.initStd.toFixed(3)}</span>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------- what to try
const TRIED = "topdown-trainlab-tried-v1";
const TRIES: { id: string; title: string; watch: string; run: (s: TrainState) => void; stay?: boolean }[] = [
  { id: "learn", title: "Watch it learn from nothing", watch: "Loss starts at 5.55 (pure guessing) and the samples go from gibberish to real words within ~200 steps.", run: () => reset(DEFAULT_CONFIG, true) },
  { id: "diverge", title: "Set the learning rate to 1", watch: "Steps get so big the loss shoots up instead of down, then explodes.", run: (s) => { setCfg({ lr: 1 }); if (s.status !== "running") start(); } },
  { id: "tiny-lr", title: "Set the learning rate to 0.0001", watch: "Nothing is wrong, it is just painfully slow. The curve barely moves.", run: (s) => { reset({ ...s.cfg, optimizer: "adamw", lr: 1e-4 }, true); } },
  { id: "no-attn", title: "Turn attention off", watch: "Tokens can no longer share information. On Tiny, 400 steps end near 1.8 instead of about 1.0, and it can't copy “red”.", run: (s) => reset({ ...s.cfg, attention: false }, true) },
  { id: "no-pos", title: "Turn positions off", watch: "The model no longer knows word order directly. Surprise: on this data it costs only a little (≈1.1 vs ≈1.0 at step 400). Why?", run: (s) => reset({ ...s.cfg, pos: false }, true) },
  { id: "no-resid", title: "Small model, no residual", watch: "Watch the gradient bars: without x + f(x), the push reaching layer 1 shrinks ~100,000× and the loss stalls near 4.5.", run: () => reset({ ...withPreset(DEFAULT_CONFIG, "small"), residual: false }, true) },
  { id: "sgd", title: "Plain SGD instead of AdamW", watch: "Same model, simpler update rule. Which gets lower after 300 steps?", run: (s) => { reset({ ...s.cfg, optimizer: "sgd", lr: DEFAULT_LR.sgd }, true); } },
  { id: "batch1", title: "Batch size 1", watch: "Each step sees one 32-token snippet: the per-step loss gets very noisy.", run: (s) => { setCfg({ batch: 1 }); if (s.status !== "running") start(); } },
  { id: "own", title: "Train on your own text", watch: "Paste a few pages of anything into Setup → Training text.", run: () => { window.dispatchEvent(new Event("trainlab-open-setup")); setTimeout(() => document.getElementById("train-text")?.scrollIntoView({ behavior: "smooth", block: "center" }), 150); }, stay: true },
];
function TryCard({ s }: { s: TrainState }) {
  const [done, setDone] = useState<Record<string, boolean>>(() => { try { return JSON.parse(localStorage.getItem(TRIED) || "{}"); } catch { return {}; } });
  const mark = (id: string, v: boolean) => { const d = { ...done, [id]: v }; setDone(d); try { localStorage.setItem(TRIED, JSON.stringify(d)); } catch { /* private mode */ } };
  const n = TRIES.filter((t) => done[t.id]).length;
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>What to try</CardTitle>
        <CardDescription>Each button sets it up and starts training. {n}/{TRIES.length} tried.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-2 @3xl/viz:grid-cols-2">
        {TRIES.map((t) => (
          <div key={t.id} className={cn("flex items-start gap-3 rounded-lg border p-3", done[t.id] && "bg-muted/40")} data-ask-id={`train-try-${t.id}`} data-ask={`Experiment idea: ${t.title}. What to watch: ${t.watch}`}>
            <Checkbox className="mt-1" checked={!!done[t.id]} onCheckedChange={(v) => mark(t.id, !!v)} aria-label={`Mark "${t.title}" as tried`} />
            <div className="min-w-0 flex-1">
              <div className={cn("text-sm font-medium", done[t.id] && "text-muted-foreground line-through decoration-muted-foreground/50")}>{t.title}</div>
              <div className="text-xs text-muted-foreground">{t.watch}</div>
            </div>
            <Button size="sm" variant="outline" className="shrink-0" onClick={() => { t.run(trainer.get()); mark(t.id, true); if (!t.stay) window.scrollTo({ top: 0, behavior: "smooth" }); }}><Play />Try</Button>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
