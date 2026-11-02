// Small shared chart + data helpers for the Backprop / Scaling / Invent labs.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { FlaskConical, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";

// ---------- data loading with polling --------------------------------------------
export type Load<T> = { status: "loading" | "missing" | "ok"; data: T | null; checked: number; retry: () => void };

/** Fetch a JSON file; if it doesn't exist yet, keep checking every `every` ms. */
export function useResearch<T>(path: string, every = 15000): Load<T> {
  const [st, setSt] = useState<{ status: Load<T>["status"]; data: T | null; checked: number }>({ status: "loading", data: null, checked: 0 });
  const [tick, setTick] = useState(0);
  const retry = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    let alive = true, timer = 0;
    const go = async () => {
      try {
        const r = await fetch(`${path}?t=${Date.now()}`, { cache: "no-store" });
        const ct = r.headers.get("content-type") ?? "";
        if (!r.ok || !ct.includes("json")) throw new Error("missing");
        const d = (await r.json()) as T;
        if (alive) setSt({ status: "ok", data: d, checked: Date.now() });
      } catch {
        if (!alive) return;
        setSt((s) => ({ ...s, status: s.data ? "ok" : "missing", checked: Date.now() }));
        timer = window.setTimeout(go, every);
      }
    };
    go();
    return () => { alive = false; clearTimeout(timer); };
  }, [path, every, tick]);
  return { ...st, retry };
}

export function Waiting({ what, file, load, className }: { what: string; file: string; load: Load<unknown>; className?: string }) {
  return (
    <Empty className={cn("border border-dashed", className)}>
      <EmptyHeader>
        <EmptyMedia variant="icon">{load.status === "loading" ? <Spinner /> : <FlaskConical />}</EmptyMedia>
        <EmptyTitle>{load.status === "loading" ? "Loading…" : `${what} are still cooking`}</EmptyTitle>
        <EmptyDescription>
          These come from a real experiment that writes <code className="font-mono text-xs">{file}</code>. This page checks again every 15 seconds.
        </EmptyDescription>
      </EmptyHeader>
      {load.status === "missing" && (
        <EmptyContent><Button variant="outline" size="sm" onClick={load.retry}><RefreshCw />Check now</Button></EmptyContent>
      )}
    </Empty>
  );
}

// ---------- sizing ---------------------------------------------------------------
export function useWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current; if (!el) return;
    setW(el.clientWidth || fallback);
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width) || fallback));
    ro.observe(el); return () => ro.disconnect();
  }, [fallback]);
  return [ref, w] as const;
}

// ---------- scales & ticks ---------------------------------------------------------
export interface Scale { (v: number): number; inv: (px: number) => number; ticks: number[]; log: boolean; domain: [number, number] }

export function scale(domain: [number, number], range: [number, number], log = false, nTicks = 5): Scale {
  const [d0, d1] = log ? [Math.log10(domain[0]), Math.log10(domain[1])] : domain;
  const span = d1 - d0 || 1;
  const f = ((v: number) => range[0] + (((log ? Math.log10(Math.max(v, 1e-300)) : v) - d0) / span) * (range[1] - range[0])) as Scale;
  f.inv = (px) => { const t = d0 + ((px - range[0]) / (range[1] - range[0])) * span; return log ? 10 ** t : t; };
  f.log = log; f.domain = domain;
  f.ticks = log ? logTicks(domain[0], domain[1], nTicks) : linTicks(domain[0], domain[1], nTicks);
  return f;
}
export function linTicks(a: number, b: number, n = 5) {
  const raw = (b - a) / n, mag = 10 ** Math.floor(Math.log10(raw || 1)), step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
export function logTicks(a: number, b: number, n = 5) {
  const lo = Math.floor(Math.log10(a)), hi = Math.ceil(Math.log10(b));
  for (const mult of [[1], [1, 3], [1, 2, 5]]) {
    const out: number[] = [];
    for (let e = lo; e <= hi; e++) for (const m of mult) { const v = m * 10 ** e; if (v >= a * 0.999 && v <= b * 1.001) out.push(v); }
    if (out.length >= Math.min(n, 3) || mult.length === 3) {
      if (out.length > n + 2) { const k = Math.ceil(out.length / (n + 1)); return out.filter((_, i) => i % k === 0); }
      return out;
    }
  }
  return [a, b];
}
export const extent = (xs: number[], pad = 0): [number, number] => {
  const lo = Math.min(...xs), hi = Math.max(...xs), p = (hi - lo) * pad;
  return [lo - p, hi + p];
};
export const logExtent = (xs: number[], padFrac = 0.15): [number, number] => {
  const lo = Math.min(...xs), hi = Math.max(...xs), r = Math.log10(hi / lo) * padFrac || 0.2;
  return [lo / 10 ** r, hi * 10 ** r];
};

/** Compact human number: 1.2k, 3.4M, 5.6B, 7.8T, 1.0e21. */
export function human(v: number, digits = 2) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e15) return v.toExponential(1).replace("e+", "e");
  const units: [number, string][] = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "k"]];
  for (const [u, s] of units) if (a >= u) return `${+(v / u).toPrecision(digits + 1)}${s}`;
  if (a < 0.01) return v.toExponential(1);
  return `${+v.toPrecision(digits + 1)}`;
}

// ---------- line chart -------------------------------------------------------------
export interface Series { id: string; label: string; color: string; points: [number, number][]; dashed?: boolean; width?: number }
export interface ChartProps {
  series: Series[];
  xLog?: boolean; yLog?: boolean;
  xDomain?: [number, number]; yDomain?: [number, number];
  xLabel: string; yLabel: string;
  xFmt?: (v: number) => string; yFmt?: (v: number) => string;
  height?: number;
  /** Extra SVG drawn on top, given the scales. */
  overlay?: (x: Scale, y: Scale) => ReactNode;
  /** Show crosshair + tooltip (hover on desktop, tap/drag on touch). */
  crosshair?: boolean;
  askId?: string; ask?: string;
  className?: string;
}

export function LineChart({ series, xLog, yLog, xDomain, yDomain, xLabel, yLabel, xFmt = human, yFmt = (v) => v.toFixed(2), height = 280, overlay, crosshair = true, askId, ask, className }: ChartProps) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hx, setHx] = useState<number | null>(null);
  const all = series.flatMap((s) => s.points).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && (!xLog || x > 0) && (!yLog || y > 0));
  if (!all.length) return <div ref={ref} className={className} />;
  const xd = xDomain ?? (xLog ? logExtent(all.map((p) => p[0]), 0.02) : extent(all.map((p) => p[0])));
  const yd = yDomain ?? (yLog ? logExtent(all.map((p) => p[1]), 0.05) : extent(all.map((p) => p[1]), 0.06));
  const narrow = W < 480;
  const m = { l: narrow ? 38 : 48, r: 12, t: 10, b: 38 };
  const H = height;
  const x = scale(xd, [m.l, W - m.r], xLog, narrow ? 4 : 6), y = scale(yd, [H - m.b, m.t], yLog, 5);
  const path = (pts: [number, number][]) => pts.filter(([a, b]) => (!xLog || a > 0) && (!yLog || b > 0) && Number.isFinite(b))
    .map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join("");
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!crosshair) return;
    const r = e.currentTarget.getBoundingClientRect(), px = e.clientX - r.left;
    setHx(px < m.l || px > W - m.r ? null : x.inv(px));
  };
  const nearest = (s: Series, xv: number) => {
    let best: [number, number] | null = null, bd = Infinity;
    for (const p of s.points) { const d = Math.abs(x(p[0]) - x(xv)); if (d < bd && Number.isFinite(p[1])) { bd = d; best = p; } }
    return bd < 40 ? best : null;
  };
  const hits = hx == null ? [] : series.map((s) => ({ s, p: nearest(s, hx) })).filter((h) => h.p);
  const tipLeft = hx != null && x(hx) > W / 2;
  return (
    <div ref={ref} className={cn("relative w-full select-none", className)} data-ask-id={askId} data-ask={ask}>
      <svg width={W} height={H} className="block touch-pan-y overflow-visible" onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={(e) => e.pointerType === "mouse" && setHx(null)}>
        {y.ticks.map((t) => (
          <g key={`y${t}`}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke="var(--viz-grid)" />
            <text x={m.l - 6} y={y(t) + 3.5} fontSize={10.5} textAnchor="end" className="fill-muted-foreground tabular-nums">{yFmt(t)}</text>
          </g>
        ))}
        {x.ticks.map((t) => (
          <g key={`x${t}`}>
            <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} stroke="var(--viz-grid)" strokeDasharray="2 3" />
            <text x={x(t)} y={H - m.b + 14} fontSize={10.5} textAnchor="middle" className="fill-muted-foreground tabular-nums">{xFmt(t)}</text>
          </g>
        ))}
        <text x={(m.l + W - m.r) / 2} y={H - 4} fontSize={11} textAnchor="middle" className="fill-muted-foreground">{xLabel}{xLog ? " (log)" : ""}</text>
        <text transform={`translate(11 ${(m.t + H - m.b) / 2}) rotate(-90)`} fontSize={11} textAnchor="middle" className="fill-muted-foreground">{yLabel}{yLog ? " (log)" : ""}</text>
        {series.map((s) => (
          <path key={s.id} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {overlay?.(x, y)}
        {hx != null && (
          <g pointerEvents="none">
            <line x1={x(hx)} x2={x(hx)} y1={m.t} y2={H - m.b} stroke="currentColor" className="text-muted-foreground" strokeOpacity={0.5} />
            {hits.map(({ s, p }) => <circle key={s.id} cx={x(p![0])} cy={y(p![1])} r={4} fill={s.color} stroke="var(--background)" strokeWidth={2} />)}
          </g>
        )}
      </svg>
      {hx != null && hits.length > 0 && (
        <div className="pointer-events-none absolute top-2 z-10 min-w-36 rounded-md border bg-popover/95 px-2.5 py-1.5 text-xs shadow-md backdrop-blur"
          style={tipLeft ? { right: W - x(hx) + 10 } : { left: x(hx) + 10 }}>
          <div className="mb-1 font-medium tabular-nums">{xLabel}: {xFmt(hits[0].p![0])}</div>
          {hits.map(({ s, p }) => (
            <div key={s.id} className="flex items-center gap-1.5 tabular-nums">
              <span className="size-2 shrink-0 rounded-full" style={{ background: s.color }} />
              <span className="flex-1 text-muted-foreground">{s.label}</span><span>{yFmt(p![1])}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function Legend({ items, className }: { items: { label: string; color: string; dashed?: boolean; dot?: boolean }[]; className?: string }) {
  return (
    <div className={cn("flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground", className)}>
      {items.map((it) => (
        <span key={it.label} className="flex items-center gap-1.5">
          {it.dot ? <span className="size-2.5 rounded-full" style={{ background: it.color }} />
            : <span className="h-0 w-4 border-t-2" style={{ borderColor: it.color, borderTopStyle: it.dashed ? "dashed" : "solid" }} />}
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** A small big-number tile. */
export function Stat({ k, v, sub, className, ...rest }: { k: string; v: ReactNode; sub?: ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("rounded-xl border bg-card px-3 py-2.5", className)} {...rest}>
      <div className="text-xs text-muted-foreground">{k}</div>
      <div className="text-xl font-semibold tabular-nums">{v}</div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}
