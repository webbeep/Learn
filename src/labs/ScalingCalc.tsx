// "Train your own" compute + memory calculator for the Scaling page.
import { useEffect, useState } from "react";
import { Calculator, Clock, Cpu, MemoryStick, Scale } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { human } from "./charts";

// TinyGPT's real run: 3000 steps × 64 sequences × 64 tokens.
const TINY_N = 170_560, TINY_D = 3000 * 64 * 64;
const MODELS = [
  { id: "tiny", name: "TinyGPT", N: TINY_N, D: TINY_D, hw: "cpu", note: "the model in this app: 3,000 steps × 64 × 64 tokens" },
  { id: "gpt2", name: "GPT-2 small", N: 124e6, D: 10e9, hw: "a100x8", note: "124M params; 10B tokens is an example budget, not GPT-2's original data" },
  { id: "gpt2xl", name: "GPT-2 XL", N: 1.5e9, D: 32e9, hw: "h100x8", note: "1.5B params; 32B tokens is an example budget" },
] as const;
const HW = [
  { id: "cpu", name: "Laptop CPU", tflops: 0.5, util: 32, gpus: 1, mem: 16, note: "rough; set so TinyGPT takes ≈ 80 s like the real run" },
  { id: "a100", name: "1× A100", tflops: 312, util: 40, gpus: 1, mem: 80, note: "312 TFLOP/s = peak dense bf16 (vendor spec)" },
  { id: "a100x8", name: "8× A100", tflops: 312, util: 40, gpus: 8, mem: 80, note: "one 8-GPU server" },
  { id: "h100x8", name: "8× H100", tflops: 989, util: 40, gpus: 8, mem: 80, note: "989 TFLOP/s = H100 SXM peak dense bf16 (vendor spec)" },
] as const;
const PREC = {
  fp32: { name: "fp32", parts: [["weights", 4], ["gradients", 4], ["Adam m + v", 8]] },
  mixed: { name: "bf16 mixed (fp32 master copy)", parts: [["weights (bf16)", 2], ["gradients (bf16)", 2], ["fp32 master weights", 4], ["Adam m + v (fp32)", 8]] },
  bf16: { name: "pure bf16", parts: [["weights", 2], ["gradients", 2], ["Adam m + v", 4]] },
} as const;
type PrecId = keyof typeof PREC;
const PART_COLORS = ["var(--viz-a)", "var(--viz-b)", "var(--viz-pin)", "color-mix(in oklch, var(--viz-a) 45%, var(--muted-foreground))"];

export function duration(sec: number) {
  if (!Number.isFinite(sec)) return "–";
  const u: [number, string][] = [[365.25 * 86400, "years"], [86400, "days"], [3600, "hours"], [60, "minutes"], [1, "seconds"]];
  for (const [k, n] of u) if (sec >= k) { const v = sec / k; return `${v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString()} ${v.toFixed(1) === "1.0" ? n.slice(0, -1) : n}`; }
  return `${(sec * 1000).toFixed(0)} ms`;
}
const bytes = (b: number) => (b >= 1e12 ? `${(b / 1e12).toFixed(1)} TB` : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} KB`);

/** Log slider + editable number for big quantities. */
function BigNumber({ id, label, value, onChange, min, max, unit }: { id: string; label: string; value: number; onChange: (v: number) => void; min: number; max: number; unit: string }) {
  const [txt, setTxt] = useState(human(value, 3));
  useEffect(() => setTxt(human(value, 3)), [value]);
  const parse = (s: string) => {
    const m = s.trim().replace(/,/g, "").match(/^([\d.]+(?:e[+-]?\d+)?)\s*([kmbt])?$/i);
    if (!m) return NaN;
    return +m[1] * ({ k: 1e3, m: 1e6, b: 1e9, t: 1e12 }[(m[2] ?? "").toLowerCase() as "k"] ?? 1);
  };
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor={id}>{label}</Label>
        <div className="flex items-center gap-1.5">
          <Input id={id} value={txt} inputMode="decimal" className="h-9 w-24 text-right font-mono tabular-nums"
            onChange={(e) => { setTxt(e.target.value); const v = parse(e.target.value); if (v > 0) onChange(v); }} onBlur={() => setTxt(human(value, 3))} />
          <span className="w-12 text-xs text-muted-foreground">{unit}</span>
        </div>
      </div>
      <Slider min={Math.log10(min)} max={Math.log10(max)} step={0.01} value={[Math.log10(value)]} onValueChange={([v]) => onChange(+(10 ** v).toPrecision(3))} aria-label={label} />
    </div>
  );
}

export function ComputeCalc({ onSummary }: { onSummary: (s: string) => void }) {
  const [model, setModel] = useState<string>("tiny");
  const [N, setN] = useState<number>(TINY_N);
  const [D, setD] = useState<number>(TINY_D);
  const [hw, setHw] = useState<string>("cpu");
  const [tflops, setTflops] = useState<number>(0.5);
  const [util, setUtil] = useState<number>(32);
  const [gpus, setGpus] = useState<number>(1);
  const [gpuMem, setGpuMem] = useState<number>(16);
  const [prec, setPrec] = useState<PrecId>("fp32");

  const pickHw = (id: string) => { const h = HW.find((x) => x.id === id)!; setHw(id); setTflops(h.tflops); setUtil(h.util); setGpus(h.gpus); setGpuMem(h.mem); setPrec(id === "cpu" ? "fp32" : "mixed"); };
  const pickModel = (id: string) => { const m = MODELS.find((x) => x.id === id)!; setModel(id); setN(m.N); setD(m.D); pickHw(m.hw); };

  const C = 6 * N * D;
  const rate = gpus * tflops * 1e12 * (util / 100);
  const secs = C / rate;
  const parts = PREC[prec].parts.map(([k, b]) => ({ k, b: b * N }));
  const total = parts.reduce((s, p) => s + p.b, 0);
  const perGpu = total / gpus;
  const ratio = D / N;
  const nOpt = Math.sqrt(C / 120), dOpt = 20 * nOpt;
  const fits = perGpu <= gpuMem * 1e9;

  useEffect(() => {
    onSummary(`Compute calculator: N = ${human(N)} params, D = ${human(D)} tokens → training compute 6·N·D = ${C.toExponential(2)} FLOPs. ` +
      `Hardware: ${gpus} × ${tflops} TFLOP/s at ${util}% utilization → ${duration(secs)}. ` +
      `Memory for weights + gradients + Adam (${PREC[prec].name}, ${total / N} bytes/param): ${bytes(total)} total, ${bytes(perGpu)} per device if sharded (activations excluded). ` +
      `Tokens per parameter: ${ratio.toFixed(1)} (Chinchilla heuristic ≈ 20). Compute-optimal for this budget ≈ ${human(nOpt)} params on ${human(dOpt)} tokens.`);
  }, [N, D, C, gpus, tflops, util, secs, prec, total, perGpu, ratio, nOpt, dOpt, onSummary]);

  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Calculator className="size-4" />Plan your own training run</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="space-y-2">
          <Label>Model preset</Label>
          <ToggleGroup type="single" variant="outline" value={model} onValueChange={(v) => v && pickModel(v)} className="grid w-full grid-cols-3">
            {MODELS.map((m) => <ToggleGroupItem key={m.id} value={m.id} className="h-auto min-h-9 flex-col gap-0 py-1.5 leading-tight"><span>{m.name}</span><span className="text-[11px] text-muted-foreground">{human(m.N)}</span></ToggleGroupItem>)}
          </ToggleGroup>
          <p className="text-xs text-muted-foreground">{MODELS.find((m) => m.id === model)?.note ?? "custom"}</p>
        </div>
        <div className="grid grid-cols-1 gap-6 @3xl:grid-cols-2 [&>*]:min-w-0">
          <div className="space-y-4">
            <BigNumber id="calc-n" label="Parameters N" value={N} onChange={(v) => { setN(v); setModel(""); }} min={1e4} max={1e12} unit="params" />
            <BigNumber id="calc-d" label="Training tokens D" value={D} onChange={(v) => { setD(v); setModel(""); }} min={1e6} max={1e14} unit="tokens" />
            <div className="space-y-2">
              <Label>Hardware</Label>
              <ToggleGroup type="single" variant="outline" value={hw} onValueChange={(v) => v && pickHw(v)} className="grid w-full grid-cols-2 @xl:grid-cols-4">
                {HW.map((h) => <ToggleGroupItem key={h.id} value={h.id} className="h-9">{h.name}</ToggleGroupItem>)}
              </ToggleGroup>
              <p className="text-xs text-muted-foreground">{HW.find((h) => h.id === hw)?.note ?? "custom"}</p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5"><Label htmlFor="calc-tf" className="text-xs">Peak TFLOP/s per device</Label>
                <Input id="calc-tf" type="number" inputMode="decimal" min={0.01} step="any" value={tflops} className="h-9 font-mono" onChange={(e) => { setTflops(Math.max(0.01, +e.target.value || 0.01)); setHw(""); }} /></div>
              <div className="space-y-1.5"><Label htmlFor="calc-g" className="text-xs">Devices</Label>
                <Input id="calc-g" type="number" inputMode="numeric" min={1} step={1} value={gpus} className="h-9 font-mono" onChange={(e) => { setGpus(Math.max(1, Math.round(+e.target.value || 1))); setHw(""); }} /></div>
              <div className="space-y-1.5"><Label htmlFor="calc-mem" className="text-xs">Memory per device (GB)</Label>
                <Input id="calc-mem" type="number" inputMode="numeric" min={1} value={gpuMem} className="h-9 font-mono" onChange={(e) => { setGpuMem(Math.max(1, +e.target.value || 1)); setHw(""); }} /></div>
              <div className="space-y-1.5"><Label className="text-xs">Precision</Label>
                <Select value={prec} onValueChange={(v) => setPrec(v as PrecId)}>
                  <SelectTrigger className="h-9 w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>{(Object.keys(PREC) as PrecId[]).map((k) => <SelectItem key={k} value={k}>{PREC[k].name}</SelectItem>)}</SelectContent>
                </Select></div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm"><Label>Utilization (MFU)</Label><span className="font-mono tabular-nums">{util}%</span></div>
              <Slider min={5} max={80} step={1} value={[util]} onValueChange={([v]) => { setUtil(v); }} aria-label="utilization" />
              <p className="text-xs text-muted-foreground">Real runs rarely hit peak: data loading, communication and memory traffic eat into it. 30–50% is a common ballpark.</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Result icon={Cpu} k="Training compute" v={`${C.toExponential(1).replace("e+", "e")} FLOPs`} sub={<span className="font-mono">6 × {human(N)} × {human(D)}</span>}
                ask={`training compute 6·N·D = 6 × ${N} × ${D} = ${C.toExponential(3)} FLOPs`} />
              <Result icon={Clock} k="Wall-clock time" v={duration(secs)} sub={`${gpus} × ${tflops} TFLOP/s × ${util}%`}
                ask={`estimated time ${duration(secs)} on ${gpus} devices at ${tflops} TFLOP/s peak and ${util}% utilization`} />
            </div>
            <div className="space-y-2 rounded-xl border p-3" data-ask-id="calc-memory" data-ask={`training memory (excluding activations): ${parts.map((p) => `${p.k} ${bytes(p.b)}`).join(", ")}; total ${bytes(total)}, per device ${bytes(perGpu)} if sharded over ${gpus}`}>
              <div className="flex items-center gap-2 text-sm font-medium"><MemoryStick className="size-4" />Memory for training state <Badge variant="secondary" className="ml-auto">{total / N} bytes / param</Badge></div>
              <div className="flex h-6 w-full gap-0.5 overflow-hidden rounded-md">
                {parts.map((p, i) => <div key={p.k} style={{ width: `${(p.b / total) * 100}%`, background: PART_COLORS[i] }} title={`${p.k}: ${bytes(p.b)}`} />)}
              </div>
              <div className="grid grid-cols-1 gap-x-4 gap-y-0.5 text-xs @md:grid-cols-2">
                {parts.map((p, i) => <div key={p.k} className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm" style={{ background: PART_COLORS[i] }} /><span className="text-muted-foreground">{p.k}</span><span className="ml-auto font-mono tabular-nums">{bytes(p.b)}</span></div>)}
              </div>
              <div className="flex flex-wrap items-baseline justify-between gap-2 border-t pt-2 text-sm">
                <span>Total <b className="font-mono">{bytes(total)}</b>{gpus > 1 && <> · <b className="font-mono">{bytes(perGpu)}</b> per device if sharded</>}</span>
                <Badge variant="outline" className={cn(fits ? "border-emerald-500/40 text-emerald-600 dark:text-emerald-400" : "border-amber-500/40 text-amber-600 dark:text-amber-400")}>{fits ? `fits in ${gpuMem} GB` : `over ${gpuMem} GB per device`}</Badge>
              </div>
              <p className="text-xs text-muted-foreground">Activations not included: they grow with batch size × context length and often dominate. Tricks like activation checkpointing trade compute for memory.</p>
            </div>
            <Chinchilla N={N} D={D} ratio={ratio} nOpt={nOpt} dOpt={dOpt} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function Result({ icon: Icon, k, v, sub, ask }: { icon: typeof Cpu; k: string; v: string; sub: React.ReactNode; ask: string }) {
  return (
    <div className="rounded-xl border bg-card p-3" data-ask-id={`calc-${k}`} data-ask={ask}>
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground"><Icon className="size-3.5" />{k}</div>
      <div className="mt-0.5 text-xl font-semibold tabular-nums">{v}</div>
      <div className="truncate text-xs text-muted-foreground">{sub}</div>
    </div>
  );
}

function Chinchilla({ N, D, ratio, nOpt, dOpt }: { N: number; D: number; ratio: number; nOpt: number; dOpt: number }) {
  const lo = -1, hi = 4; // 0.1 … 10,000 tokens per param, log scale
  const pos = (r: number) => `${Math.max(0, Math.min(100, ((Math.log10(r) - lo) / (hi - lo)) * 100))}%`;
  const verdict = ratio < 10 ? "fewer tokens than the heuristic suggests: likely under-trained for its size" : ratio > 40 ? "more tokens than the heuristic: a smaller model is being trained longer (common when cheap inference matters)" : "close to the compute-optimal heuristic";
  return (
    <div className="space-y-2 rounded-xl border p-3" data-ask-id="calc-chinchilla" data-ask={`tokens per parameter ${ratio.toFixed(1)} vs Chinchilla heuristic ≈ 20; compute-optimal for same compute ≈ ${human(nOpt)} params, ${human(dOpt)} tokens`}>
      <div className="flex items-center gap-2 text-sm font-medium"><Scale className="size-4" />Tokens per parameter: <span className="font-mono tabular-nums">{ratio < 10 ? ratio.toFixed(1) : Math.round(ratio).toLocaleString()}</span></div>
      <div className="relative h-8">
        <div className="absolute inset-x-0 top-3 h-2 rounded-full bg-muted" />
        <div className="absolute top-2 h-4 w-px bg-foreground/60" style={{ left: pos(20) }} />
        <div className="absolute -top-1.5 translate-x-1 text-[10px] text-muted-foreground" style={{ left: pos(20) }}>≈20</div>
        <div className="absolute top-2 size-4 -translate-x-1/2 rounded-full border-2 border-background bg-viz-a" style={{ left: pos(ratio) }} />
        <div className="absolute inset-x-0 top-6 flex justify-between text-[10px] text-muted-foreground"><span>0.1</span><span>10,000</span></div>
      </div>
      <p className="text-xs text-muted-foreground">{N && D ? <>Yours is {verdict}. For the same compute, the rule of thumb points at ≈ <b className="text-foreground">{human(nOpt)}</b> params on <b className="text-foreground">{human(dOpt)}</b> tokens.</> : null}</p>
      <p className="text-[11px] text-muted-foreground">A rough heuristic from Hoffmann et al. 2022 (<a className="underline" href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noreferrer">"Chinchilla"</a>): for a fixed compute budget, scale parameters and tokens together, about 20 tokens per parameter. The exact ratio depends on data, architecture and setup.</p>
    </div>
  );
}
