import { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Cpu, FlaskConical, TriangleAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { setScreen } from "../lib/store";
import { pct, show } from "../lib/viz";
import { LineChart, catColor } from "../train/charts";
import { trainHref } from "../train/config";

interface Run {
  id: string; label: string; what: string; config: Record<string, number | boolean | string>;
  params: number; curve: { step: number[]; train: (number | null)[]; val: (number | null)[] };
  final_val: number | null; final_train?: number | null; diverged: boolean; seconds?: number;
  samples: { prompt: string; text: string }[]; probe_top: { token: string; p: number }[];
}
interface Ablations { steps: number; batch: number; tokens_per_step?: number; note: string; probe_prompt: string; runs: Run[] }

const fmtP = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : `${(n / 1e3).toFixed(n < 1e4 ? 1 : 0)}k`);
const lossTxt = (v: number | null | undefined) => (v == null || !isFinite(v) ? "—" : v.toFixed(3));

/** A browser-sized version of a research config, as a #/train link. */
function retrainLink(r: Run) {
  const c = r.config, L = Number(c.n_layer ?? 3);
  const q: Record<string, unknown> = { preset: "small", layers: L <= 1 ? 1 : L >= 6 ? 4 : 2 };
  for (const k of ["attention", "mlp", "positions", "residual", "layernorm"]) if (c[`no_${k}`] !== undefined) q[k] = !c[`no_${k}`];
  return trainHref(q);
}

export function ArchLab() {
  const [data, setData] = useState<Ablations | null>(null);
  const [failed, setFailed] = useState(false);
  const [on, setOn] = useState<string[]>([]);
  const [zoom, setZoom] = useState<"all" | "zoom">("zoom");
  const [sort, setSort] = useState<{ key: "label" | "final_val" | "params" | "delta"; dir: 1 | -1 }>({ key: "final_val", dir: 1 });
  const [a, setA] = useState("baseline"), [b, setB] = useState("no-attention");

  useEffect(() => {
    let alive = true, timer = 0;
    const load = () => fetch("/research/ablations.json", { cache: "no-cache" })
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then((d: Ablations) => {
        if (!alive) return;
        setData(d); setFailed(false);
        setOn((cur) => cur.length ? cur : d.runs.filter((r) => ["baseline", "no-attention", "no-mlp", "no-layernorm"].includes(r.id)).map((r) => r.id).concat(d.runs.length < 3 ? d.runs.map((r) => r.id) : []));
        setA((x) => d.runs.some((r) => r.id === x) ? x : d.runs[0]?.id ?? "");
        setB((x) => d.runs.some((r) => r.id === x) ? x : d.runs[1]?.id ?? d.runs[0]?.id ?? "");
      })
      .catch(() => { if (alive) { setFailed(true); timer = window.setTimeout(load, 30000); } }); // results may still be cooking
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  const runs = data?.runs ?? [];
  const color = (id: string) => catColor(Math.max(0, runs.findIndex((r) => r.id === id)));
  const base = runs.find((r) => r.id === "baseline") ?? runs[0];
  const delta = (r: Run) => (r.final_val != null && base?.final_val != null ? r.final_val - base.final_val : null);
  const sorted = useMemo(() => {
    const v = (r: Run) => sort.key === "label" ? r.label : sort.key === "params" ? r.params : sort.key === "delta" ? (delta(r) ?? Infinity) : (r.final_val ?? Infinity);
    return [...runs].sort((x, y) => { const p = v(x), q = v(y); return (p < q ? -1 : p > q ? 1 : 0) * sort.dir; });
  }, [runs, sort]); // eslint-disable-line

  useEffect(() => {
    if (!data) { setScreen("Architecture lab", failed ? "results not published yet" : "loading", "The architecture ablation results (public/research/ablations.json) are not available yet."); return; }
    const ra = runs.find((r) => r.id === a), rb = runs.find((r) => r.id === b);
    setScreen("Architecture lab", `Comparing ${on.length} runs`, [
      `Ablation study: each variant of TinyGPT trained from the same seed for ${data.steps} steps of batch ${data.batch}. ${data.note}`,
      `Results (final validation loss, params): ${runs.map((r) => `${r.label} ${lossTxt(r.final_val)}${r.diverged ? " (diverged)" : ""}, ${r.params.toLocaleString()} params`).join("; ")}.`,
      `Curves shown: ${on.map((id) => runs.find((r) => r.id === id)?.label).join(", ")}.`,
      ra && rb ? `Side by side: "${ra.label}" vs "${rb.label}". Probe "${data.probe_prompt}" → ${ra.label}: ${ra.probe_top.map((t) => `"${t.token}" ${pct(t.p)}`).join(", ")}; ${rb.label}: ${rb.probe_top.map((t) => `"${t.token}" ${pct(t.p)}`).join(", ")}.` : "",
      ra && rb ? `Samples: ${[ra, rb].map((r) => `${r.label}: ${r.samples.map((s) => `"${s.prompt}" → "${s.text}"`).join(" | ")}`).join(" ;; ")}` : "",
    ].filter(Boolean).join("\n"));
  }, [data, failed, on.join(), a, b]); // eslint-disable-line

  const header = (
    <div>
      <h1 className="text-3xl font-semibold tracking-tight">What does each part do?</h1>
      <p className="mt-1 max-w-3xl text-muted-foreground">Take TinyGPT, remove one part, retrain from the same random start on the same data, compare. Every number here comes from a real training run.</p>
    </div>
  );
  if (!data) return (
    <div className="space-y-6 pb-10">
      {header}
      {failed ? (
        <Empty className="border border-dashed">
          <EmptyHeader>
            <EmptyMedia variant="icon"><FlaskConical /></EmptyMedia>
            <EmptyTitle>The experiments are still running</EmptyTitle>
            <EmptyDescription>Results appear here as soon as <code>train/experiments/ablations.py</code> publishes them. This page checks again every 30 seconds. Meanwhile, run your own ablations live.</EmptyDescription>
          </EmptyHeader>
          <Button asChild><a href="#/train"><Cpu />Train it yourself</a></Button>
        </Empty>
      ) : <div className="space-y-4"><Skeleton className="h-10 w-full" /><Skeleton className="h-72 w-full" /><Skeleton className="h-48 w-full" /></div>}
    </div>
  );

  // chart
  const sel = runs.filter((r) => on.includes(r.id));
  const late = sel.flatMap((r) => r.curve.val.filter((_, i) => r.curve.step[i] >= data.steps * 0.33)).filter((v): v is number => v != null && isFinite(v));
  const zMin = late.length ? Math.max(0, Math.floor((Math.min(...late) - 0.05) * 10) / 10) : 0;
  const zMax = late.length ? Math.ceil((Math.max(...late) + 0.05) * 10) / 10 : 6;
  const zoomed = zoom === "zoom" && late.length > 0;
  const ra = runs.find((r) => r.id === a), rb = runs.find((r) => r.id === b);
  const pMax = Math.max(0.05, ...runs.flatMap((r) => r.probe_top.map((t) => t.p))); // one shared scale so runs compare honestly
  const redBest = Math.max(0, ...runs.map((r) => r.probe_top.find((t) => t.token === " red")?.p ?? 0));
  const sortBtn = (key: typeof sort.key, label: string, cls = "") => (
    <button className={cn("inline-flex items-center gap-1 font-medium hover:text-foreground", sort.key === key && "text-foreground", cls)}
      onClick={() => setSort((s) => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : 1 }))}>
      {label}{sort.key === key ? (sort.dir === 1 ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />) : <ArrowUpDown className="size-3 opacity-40" />}
    </button>
  );

  return (
    <div className="@container space-y-6 pb-10">
      {header}

      <Card className="gap-3">
        <CardHeader>
          <CardTitle>Validation loss while training</CardTitle>
          <CardDescription>{data.steps.toLocaleString()} steps × {data.batch} sequences{data.tokens_per_step ? ` (${data.tokens_per_step.toLocaleString()} tokens/step)` : ""}. Lower is better. Tap a run to show or hide it.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            {runs.map((r) => {
              const active = on.includes(r.id);
              return (
                <button key={r.id} onClick={() => setOn((cur) => active ? cur.filter((x) => x !== r.id) : [...cur, r.id])} aria-pressed={active}
                  className={cn("inline-flex h-9 items-center gap-2 rounded-full border px-3 text-sm transition-colors", active ? "bg-muted font-medium" : "text-muted-foreground opacity-70 hover:opacity-100")}>
                  <span className="size-2.5 rounded-full" style={{ background: active ? color(r.id) : "transparent", boxShadow: `inset 0 0 0 1.5px ${color(r.id)}` }} />
                  {r.label}{r.diverged && <TriangleAlert className="size-3.5 text-destructive" />}
                </button>
              );
            })}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex gap-1.5">
              <Button size="sm" variant="ghost" className="h-8" onClick={() => setOn(runs.map((r) => r.id))}>All</Button>
              <Button size="sm" variant="ghost" className="h-8" onClick={() => setOn(base ? [base.id] : [])}>Baseline only</Button>
            </div>
            <ToggleGroup type="single" size="sm" variant="outline" value={zoom} onValueChange={(v) => v && setZoom(v as "all" | "zoom")}>
              <ToggleGroupItem value="zoom" className="h-8 px-3">Zoom on the finish</ToggleGroupItem>
              <ToggleGroupItem value="all" className="h-8 px-3">Whole run</ToggleGroupItem>
            </ToggleGroup>
          </div>
          {sel.length ? (
            <LineChart askId="lab-curves" height={300}
              ask={`Validation loss curves of the ablation runs shown: ${sel.map((r) => `${r.label} final ${lossTxt(r.final_val)}`).join(", ")}${zoomed ? ` (zoomed to steps ≥ ${Math.round(data.steps * 0.33)}, loss ${zMin}–${zMax})` : ""}.`}
              xMin={zoomed ? Math.round(data.steps * 0.33 / 50) * 50 : 0} xMax={data.steps} yMin={zoomed ? zMin : 0} yMax={zoomed ? zMax : 6}
              refs={zoomed ? [] : [{ y: Math.log(256), label: "random guessing (ln 256)" }]}
              series={sel.map((r) => ({ id: r.id, label: r.label, color: color(r.id), x: r.curve.step, y: r.curve.val, width: r.id === "baseline" ? 2.6 : 1.8, dashed: r.id === "baseline" }))}
              fmtY={(v) => v.toFixed(3)} endLabels />
          ) : <div className="flex h-40 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">Pick at least one run above</div>}
          {zoomed && sel.some((r) => (r.final_val ?? 0) > zMax) && <p className="text-xs text-muted-foreground">Some curves end above the zoomed window. Switch to “Whole run” to see them.</p>}
        </CardContent>
      </Card>

      <Card className="gap-3">
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>Final validation loss, and how much worse than the full model. Tap a column to sort.</CardDescription>
        </CardHeader>
        <CardContent className="px-0 sm:px-6">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[22rem] text-sm">
              <thead className="text-xs text-muted-foreground">
                <tr className="border-b">
                  <th className="py-2 pr-2 pl-4 text-left sm:pl-0">{sortBtn("label", "Run")}</th>
                  <th className="px-2 py-2 text-right">{sortBtn("final_val", "Val loss", "justify-end whitespace-nowrap")}</th>
                  <th className="px-2 py-2 text-right">{sortBtn("delta", "vs full", "justify-end whitespace-nowrap")}</th>
                  <th className="hidden px-2 py-2 text-right @lg:table-cell">{sortBtn("params", "Params", "justify-end")}</th>
                  <th className="py-2 pr-4 pl-2 text-right sm:pr-0"><span className="sr-only">Retrain</span></th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => {
                  const d = delta(r), worst = Math.max(...runs.map((x) => delta(x) ?? 0), 0.01);
                  return (
                    <tr key={r.id} className="border-b last:border-0 align-top" data-ask-id={`lab-row-${r.id}`}
                      data-ask={`Ablation "${r.label}": ${r.what} Final validation loss ${lossTxt(r.final_val)} (train ${lossTxt(r.final_train)}), ${d == null ? "" : `${d >= 0 ? "+" : ""}${d.toFixed(3)} vs the full model, `}${r.params.toLocaleString()} parameters${r.diverged ? ", DIVERGED" : ""}${r.seconds ? `, trained in ${r.seconds}s on CPU` : ""}.`}>
                      <td className="py-2.5 pr-2 pl-4 sm:pl-0">
                        <div className="flex items-center gap-2 font-medium"><span className="size-2.5 shrink-0 rounded-full" style={{ background: color(r.id) }} />{r.label}{r.diverged && <Badge variant="destructive" className="h-5">diverged</Badge>}</div>
                        <div className="mt-0.5 max-w-md pl-[18px] text-xs text-muted-foreground">{r.what}</div>
                      </td>
                      <td className="px-2 py-2.5 text-right font-semibold tabular-nums">{lossTxt(r.final_val)}</td>
                      <td className="px-2 py-2.5 text-right tabular-nums">
                        {d == null ? "—" : r.id === base?.id ? <span className="text-muted-foreground">baseline</span> : (
                          <div className="flex flex-col items-end gap-1">
                            <span className={cn(d > 0.05 ? "text-destructive" : d < -0.005 ? "text-viz-b" : "text-muted-foreground")}>{d >= 0 ? "+" : ""}{d.toFixed(3)}</span>
                            <span className="h-1 w-16 rounded-full bg-muted"><span className="block h-full rounded-full bg-destructive/70" style={{ width: `${Math.max(0, Math.min(1, d / worst)) * 100}%` }} /></span>
                          </div>
                        )}
                      </td>
                      <td className="hidden px-2 py-2.5 text-right text-muted-foreground tabular-nums @lg:table-cell">{fmtP(r.params)}</td>
                      <td className="py-2 pr-4 pl-2 text-right sm:pr-0">
                        <Button size="sm" variant="outline" asChild title="Open this setup in the Train lab (a smaller, browser-sized model)">
                          <a href={retrainLink(r)}><Cpu /><span className="hidden @md:inline">Retrain</span></a>
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 px-4 text-xs text-muted-foreground sm:px-0">“Retrain” opens the same switch in the Train lab on a smaller model (d 48, up to 4 layers) so it trains in seconds in your browser.</p>
        </CardContent>
      </Card>

      <Card className="gap-3">
        <CardHeader>
          <CardTitle>Side by side</CardTitle>
          <CardDescription>Same prompts, same sampling seed. Only the architecture differs.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 @3xl:grid-cols-2">
            {([[a, setA], [b, setB]] as const).map(([id, set], k) => {
              const r = runs.find((x) => x.id === id);
              return (
                <div key={k} className="min-w-0 space-y-3">
                  <Select value={id} onValueChange={set}>
                    <SelectTrigger className="h-10 w-full"><SelectValue /></SelectTrigger>
                    <SelectContent>{runs.map((x) => <SelectItem key={x.id} value={x.id}><span className="size-2.5 rounded-full" style={{ background: color(x.id) }} />{x.label}</SelectItem>)}</SelectContent>
                  </Select>
                  {r && <>
                    <p className="text-xs text-muted-foreground">{r.what} <span className="font-medium text-foreground tabular-nums">Val loss {lossTxt(r.final_val)}.</span></p>
                    <ProbeBars r={r} prompt={data.probe_prompt} color={color(r.id)} max={pMax} />
                    <div className="space-y-2">
                      {r.samples.map((s, i) => (
                        <div key={i} className="rounded-lg border bg-muted/40 p-2.5 font-mono text-[13px] leading-relaxed break-words whitespace-pre-wrap"
                          data-ask-id={`lab-sample-${r.id}-${i}`} data-ask={`The "${r.label}" model (val loss ${lossTxt(r.final_val)}), given "${s.prompt}", wrote "${s.text}" (temperature 0.8).`}>
                          <span className="text-muted-foreground">{s.prompt}</span>{s.text}
                        </div>
                      ))}
                    </div>
                  </>}
                </div>
              );
            })}
          </div>
          {ra && rb && <p className="text-xs text-muted-foreground">Showing <b className="text-foreground">{ra.label}</b> vs <b className="text-foreground">{rb.label}</b>.</p>}
        </CardContent>
      </Card>

      <Card className="gap-3">
        <CardHeader>
          <CardTitle>Every run on the test sentence</CardTitle>
          <CardDescription><span className="font-mono text-xs">{data.probe_prompt} <span className="text-pin">___</span></span>
            {redBest < 0.5 && <span className="mt-1 block">The right answer is “red”. After {data.steps.toLocaleString()} steps no variant has learned to copy it yet (best: {pct(redBest)}); the full 3,000-step TinyGPT gives it 87%. Bars share one scale.</span>}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 @lg:grid-cols-2 @4xl:grid-cols-4">
          {runs.map((r) => (
            <div key={r.id} className="rounded-lg border p-3">
              <div className="mb-2 flex items-center gap-2 text-sm font-medium"><span className="size-2.5 rounded-full" style={{ background: color(r.id) }} />{r.label}</div>
              <ProbeBars r={r} prompt={data.probe_prompt} color={color(r.id)} max={pMax} compact />
            </div>
          ))}
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">{data.note}</p>
    </div>
  );
}

function ProbeBars({ r, prompt, color, compact, max }: { r: Run; prompt: string; color: string; compact?: boolean; max: number }) {
  return (
    <div className="space-y-1" data-ask-id={`lab-probe-${r.id}${compact ? "-grid" : ""}`}
      data-ask={`After "${prompt}", the "${r.label}" model's top-5 next tokens: ${r.probe_top.map((t) => `"${t.token}" ${pct(t.p)}`).join(", ")}. (The right answer is " red".)`}>
      {!compact && <div className="text-xs text-muted-foreground">Next word after “…now tom has the”</div>}
      {r.probe_top.map((t) => (
        <div key={t.token} className="grid grid-cols-[4.5rem_1fr_2.75rem] items-center gap-2 text-xs">
          <span className={cn("truncate font-mono", t.token === " red" && "font-semibold text-pin")}>{show(t.token)}</span>
          <div className="h-2.5 rounded-sm bg-muted"><div className="h-full rounded-sm" style={{ width: `${(t.p / max) * 100}%`, background: t.token === " red" ? "var(--viz-pin)" : color }} /></div>
          <span className="text-right tabular-nums">{pct(t.p)}</span>
        </div>
      ))}
    </div>
  );
}
