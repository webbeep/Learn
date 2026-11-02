// "Now in the real model": real gradients of TinyGPT from public/research/gradients.json.
import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { div } from "../lib/viz";
import { Legend, Stat, Waiting, useResearch } from "./charts";

interface Tensor { name: string; group?: string; shape: number[]; grad_norm: number; param_norm: number; grad_norm_step0?: number; param_norm_step0?: number; numel?: number; grad_rms?: number; grad_rms_step0?: number }
interface Grads {
  note?: string; batch?: { size: number; n_ctx: number; loss_trained: number; loss_step0: number };
  per_tensor: Tensor[];
  per_layer_norm_at: { layers?: string[]; step0: number[]; trained: number[] };
  example: { prompt: string; target: string; loss: number; p_target?: number; dloss_dlogit_top: { token: string; grad: number; p?: number }[]; explain?: string };
}

const groupOf = (t: Tensor) => t.group ?? (t.name.startsWith("blocks.") ? `block ${+t.name.split(".")[1] + 1}` : t.name.startsWith("ln_f") ? "final LayerNorm" : "embeddings");
const pretty = (name: string) => {
  const n = name.replace(/^blocks\.\d+\./, "");
  const map: Record<string, string> = { "tok.weight": "token embeddings", "pos.weight": "position embeddings", "attn_qkv.weight": "attention Q,K,V", "attn_out.weight": "attention out", "mlp_in.weight": "MLP in", "mlp_out.weight": "MLP out" };
  return map[n] ?? n.replace(".weight", " scale").replace(".bias", " bias").replace("ln_f", "LayerNorm").replace("attn_", "attn ").replace("mlp_", "MLP ");
};
const show = (t: string) => t.replace(/ /g, "␣").replace(/\n/g, "↵");

export function RealGrads({ onSummary }: { onSummary: (s: string) => void }) {
  const load = useResearch<Grads>("/research/gradients.json");
  const d = load.data;
  useEffect(() => {
    if (!d) { onSummary("Real-model gradients section: waiting for public/research/gradients.json."); return; }
    const pl = d.per_layer_norm_at, layers = pl.layers ?? pl.trained.map((_, i) => `group ${i + 1}`);
    onSummary([
      `Real TinyGPT gradients on one batch${d.batch ? ` (${d.batch.size}×${d.batch.n_ctx} tokens; loss ${d.batch.loss_step0.toFixed(2)} at init vs ${d.batch.loss_trained.toFixed(2)} trained)` : ""}.`,
      `Gradient norm per layer, init → trained: ${layers.map((l, i) => `${l} ${pl.step0[i]?.toFixed(3)} → ${pl.trained[i]?.toFixed(3)}`).join(", ")}.`,
      `Example "${d.example.prompt}" → target "${d.example.target}", loss ${d.example.loss.toFixed(3)}. dLoss/dlogit (p − onehot): ${d.example.dloss_dlogit_top.map((g) => `"${g.token}" ${g.grad.toFixed(3)}`).join(", ")}.`,
    ].join("\n"));
  }, [d, onSummary]);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Now in the real model</h2>
        <p className="mt-1 max-w-3xl text-muted-foreground">TinyGPT is the same idea with 170,560 leaves instead of 9. These are its real gradients on one batch of training text: once at random init, once fully trained.</p>
      </div>
      {!d ? <Waiting what="Real gradients" file="public/research/gradients.json" load={load} /> : <>
        {d.batch && (
          <div className="grid grid-cols-2 gap-3 @container md:grid-cols-4">
            <Stat k="loss at init" v={d.batch.loss_step0.toFixed(2)} sub="≈ ln 256: pure guessing" />
            <Stat k="loss trained" v={d.batch.loss_trained.toFixed(2)} sub="same batch" />
            <Stat k="tensors" v={d.per_tensor.length} sub="each gets a gradient" />
            <Stat k="numbers" v={d.per_tensor.reduce((s, t) => s + (t.numel ?? t.shape.reduce((a, b) => a * b, 1)), 0).toLocaleString()} sub="one gradient each" />
          </div>
        )}
        <div className="grid gap-4 lg:grid-cols-5">
          <LogitGrads ex={d.example} />
          <LayerNorms pl={d.per_layer_norm_at} />
        </div>
        <TensorBars tensors={d.per_tensor} />
      </>}
    </div>
  );
}

function LogitGrads({ ex }: { ex: Grads["example"] }) {
  const rows = ex.dloss_dlogit_top, mx = Math.max(...rows.map((r) => Math.abs(r.grad)));
  return (
    <Card className="@container lg:col-span-3">
      <CardHeader>
        <CardTitle>Where backprop starts: ∂loss/∂logit</CardTitle>
        <CardDescription>For the last token, the gradient on each of the 256 logits is just <span className="font-mono">p − onehot</span>: the right answer gets pulled up, every other token gets pushed down by its probability.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="rounded-lg border bg-muted/40 p-3 font-mono text-[13px] leading-relaxed" data-ask-id="rg-prompt" data-ask={`prompt "${ex.prompt}" with correct next token "${ex.target}", loss ${ex.loss.toFixed(3)}`}>
          <span className="text-muted-foreground">{ex.prompt}</span>
          <span className="ml-1 rounded bg-viz-a/20 px-1 text-foreground">{show(ex.target)}</span>
          <div className="mt-1 font-sans text-xs text-muted-foreground">target {show(ex.target)}{ex.p_target != null && ` · model gives it ${(ex.p_target * 100).toFixed(1)}%`} · loss −ln p = {ex.loss.toFixed(3)}</div>
        </div>
        <div className="space-y-1">
          {rows.map((r) => {
            const w = (Math.abs(r.grad) / mx) * 50;
            return (
              <div key={r.token} className="grid grid-cols-[5.5rem_1fr_4.5rem] items-center gap-2 text-sm" data-ask-id={`rg-logit-${r.token}`}
                data-ask={`dLoss/dlogit for token "${r.token}" = ${r.grad.toFixed(4)}${r.p != null ? ` (probability ${(r.p * 100).toFixed(2)}%)` : ""}`}>
                <span className={cn("truncate font-mono text-[13px]", r.token === ex.target && "font-semibold")}>{show(r.token)}</span>
                <div className="relative h-5">
                  <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
                  <div className="absolute inset-y-0.5 rounded-sm" style={{ background: div(Math.sign(r.grad), 1), width: `max(${w}%, 2px)`, ...(r.grad < 0 ? { right: "50%" } : { left: "50%" }) }} />
                </div>
                <span className="text-right font-mono text-xs tabular-nums">{r.grad >= 0 ? "+" : "−"}{Math.abs(r.grad).toFixed(3)}</span>
              </div>
            );
          })}
        </div>
        <div className="flex justify-between text-xs text-muted-foreground"><span>← negative: raise this logit</span><span>positive: lower it →</span></div>
        <p className="text-xs text-muted-foreground">Same as the neuron above: the gradient says which way to nudge. Every number in the network then gets its share via the chain rule.</p>
      </CardContent>
    </Card>
  );
}

function LayerNorms({ pl }: { pl: Grads["per_layer_norm_at"] }) {
  const layers = pl.layers ?? pl.trained.map((_, i) => `group ${i + 1}`);
  const mx = Math.max(...pl.step0, ...pl.trained);
  return (
    <Card className="lg:col-span-2">
      <CardHeader>
        <CardTitle>Gradient size per layer</CardTitle>
        <CardDescription>Length of the gradient vector for each part of the model. Big at init (it's very wrong); small once trained (little left to fix).</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {layers.map((l, i) => (
          <div key={l} className="space-y-1" data-ask-id={`rg-layer-${i}`} data-ask={`gradient norm of ${l}: ${pl.step0[i].toFixed(4)} at init, ${pl.trained[i].toFixed(4)} when trained`}>
            <div className="text-xs text-muted-foreground">{l}</div>
            {([["init", pl.step0[i], "var(--muted-foreground)"], ["trained", pl.trained[i], "var(--viz-a)"]] as const).map(([k, v, c]) => (
              <div key={k} className="flex items-center gap-2">
                <div className="h-3 flex-1"><div className="h-full rounded-r-sm" style={{ width: `max(${(v / mx) * 100}%, 2px)`, background: c, opacity: k === "init" ? 0.45 : 1 }} /></div>
                <span className="w-12 text-right font-mono text-xs tabular-nums">{v.toFixed(3)}</span>
              </div>
            ))}
          </div>
        ))}
        <Legend items={[{ label: "random init (step 0)", color: "color-mix(in oklch, var(--muted-foreground) 45%, transparent)" }, { label: "trained (step 3000)", color: "var(--viz-a)" }]} />
      </CardContent>
    </Card>
  );
}

function TensorBars({ tensors }: { tensors: Tensor[] }) {
  const [when, setWhen] = useState<"trained" | "step0">("trained");
  const [metric, setMetric] = useState<"norm" | "ratio">("norm");
  const [all, setAll] = useState(false);
  const hasStep0 = tensors.some((t) => t.grad_norm_step0 != null);
  const val = (t: Tensor) => {
    const g = when === "step0" ? t.grad_norm_step0 ?? t.grad_norm : t.grad_norm;
    const p = when === "step0" ? t.param_norm_step0 ?? t.param_norm : t.param_norm;
    return metric === "norm" ? g : g / Math.max(p, 1e-12);
  };
  const shown = tensors.filter((t) => all || (t.shape.length === 2));
  const vals = shown.map(val).filter((v) => v > 0);
  const lo = Math.min(...vals) / 1.5, hi = Math.max(...vals) * 1.2;
  const pos = (v: number) => (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  const groups = [...new Set(shown.map(groupOf))];
  const top = shown.reduce((a, b) => (val(b) > val(a) ? b : a));
  const fmtV = (v: number) => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1));
  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle>Every weight matrix gets a gradient</CardTitle>
        <CardDescription>One bar per tensor, grouped by layer (log scale). {metric === "ratio" ? "Gradient ÷ weight size: how big a nudge is relative to the numbers it nudges." : "Length of each tensor's gradient vector."}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col gap-3 @2xl:flex-row @2xl:items-center">
          {hasStep0 && (
            <ToggleGroup type="single" variant="outline" value={when} onValueChange={(v) => v && setWhen(v as typeof when)} className="w-full @2xl:w-auto">
              <ToggleGroupItem value="step0" className="h-9 flex-1 px-3">At init</ToggleGroupItem>
              <ToggleGroupItem value="trained" className="h-9 flex-1 px-3">Trained</ToggleGroupItem>
            </ToggleGroup>
          )}
          <ToggleGroup type="single" variant="outline" value={metric} onValueChange={(v) => v && setMetric(v as typeof metric)} className="w-full @2xl:w-auto">
            <ToggleGroupItem value="norm" className="h-9 flex-1 px-3">‖grad‖</ToggleGroupItem>
            <ToggleGroupItem value="ratio" className="h-9 flex-1 px-3">‖grad‖ / ‖weight‖</ToggleGroupItem>
          </ToggleGroup>
          <div className="flex items-center gap-2 @2xl:ml-auto"><Switch id="rg-all" checked={all} onCheckedChange={setAll} /><Label htmlFor="rg-all" className="text-sm">Biases &amp; LayerNorm too ({tensors.length} tensors)</Label></div>
        </div>
        <div className="grid gap-x-8 gap-y-4 @3xl:grid-cols-2">
          {groups.map((g) => (
            <div key={g} className="space-y-1">
              <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{g}</div>
              {shown.filter((t) => groupOf(t) === g).map((t) => {
                const v = val(t);
                return (
                  <div key={t.name} className="grid grid-cols-[8.5rem_1fr_3.5rem] items-center gap-2 text-sm" data-ask-id={`rg-t-${t.name}`}
                    data-ask={`${t.name} shape [${t.shape.join("×")}]: gradient norm ${t.grad_norm.toExponential(2)} trained${t.grad_norm_step0 != null ? `, ${t.grad_norm_step0.toExponential(2)} at init` : ""}; weight norm ${t.param_norm.toFixed(2)}`}>
                    <span className="truncate text-[13px]" title={t.name}>{pretty(t.name)} <span className="text-[11px] text-muted-foreground">{t.shape.join("×")}</span></span>
                    <div className="relative h-4 rounded-sm bg-muted/50"><div className="h-full rounded-r-sm bg-viz-a transition-[width] duration-500" style={{ width: `${Math.max(1, pos(v) * 100)}%` }} /></div>
                    <span className="text-right font-mono text-xs tabular-nums">{fmtV(v)}</span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">Bars use a log scale from {fmtV(lo)} to {fmtV(hi)}, so a bar twice as long is many times bigger. Biggest right now: <b className="text-foreground">{pretty(top.name)}</b> ({groupOf(top)}).</p>
      </CardContent>
    </Card>
  );
}
