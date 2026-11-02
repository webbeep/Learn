// Small responsive SVG charts shared by the Train lab and the Architecture lab.
import { useLayoutEffect, useRef, useState } from "react";
import { isDark } from "@/lib/theme";

/** Width of an element, tracked with ResizeObserver (charts draw in real pixels so text stays legible on phones). */
export function useWidth<T extends HTMLElement>(fallback = 600) {
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

/** Fixed-order categorical palette (validated light/dark steps), assigned by run, never by rank. */
const CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const catColor = (i: number) => (isDark() ? CAT_DARK : CAT_LIGHT)[i % 8];

export interface Series { id: string; label: string; color: string; x: number[]; y: (number | null)[]; dashed?: boolean; width?: number; opacity?: number; dots?: boolean; hideInTip?: boolean }
export interface RefLine { y: number; label: string }

const niceStep = (span: number, target: number) => {
  const raw = span / target, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
};
export const fmtStep = (s: number) => (s >= 10000 ? `${(s / 1000).toFixed(0)}k` : s >= 1000 ? `${(s / 1000).toFixed(s % 1000 ? 1 : 0)}k` : String(s));

/**
 * Line chart with a tap/hover crosshair. Values above yMax are drawn clipped at the top edge.
 * askId/ask: the whole chart is clickable for the tutor; each crosshair position is too.
 */
export function LineChart({ series, xMax, yMax, yMin = 0, refs = [], height = 260, xLabel = "step", askId, ask, fmtY = (v) => v.toFixed(2) }: {
  series: Series[]; xMax: number; yMax: number; yMin?: number; refs?: RefLine[]; height?: number; xLabel?: string;
  askId: string; ask: string; fmtY?: (v: number) => string;
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hx, setHx] = useState<number | null>(null);
  const H = height, L = 38, R = 12, T = 10, B = 28, iw = Math.max(10, W - L - R), ih = H - T - B;
  const X = (x: number) => L + (x / Math.max(1, xMax)) * iw;
  const Y = (y: number) => T + (1 - (Math.min(yMax, Math.max(yMin, y)) - yMin) / (yMax - yMin)) * ih;
  const yStep = niceStep(yMax - yMin, 5), xStep = niceStep(Math.max(1, xMax), Math.max(2, Math.floor(iw / 80)));
  const yTicks: number[] = []; for (let v = Math.ceil(yMin / yStep) * yStep; v <= yMax + 1e-9; v += yStep) yTicks.push(+v.toFixed(6));
  const xTicks: number[] = []; for (let v = 0; v <= xMax + 1e-9; v += xStep) xTicks.push(v);
  const path = (s: Series) => {
    let d = "", pen = false;
    s.x.forEach((x, i) => { const y = s.y[i]; if (y == null || !isFinite(y)) { pen = false; return; } d += `${pen ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`; pen = true; });
    return d;
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect(), px = e.clientX - r.left;
    setHx(Math.max(0, Math.min(xMax, ((px - L) / iw) * xMax)));
  };
  // nearest point per series at the crosshair
  const tip = hx == null ? [] : series.filter((s) => !s.hideInTip && s.x.length).map((s) => {
    let bi = 0, bd = Infinity; s.x.forEach((x, i) => { const d = Math.abs(x - hx); if (d < bd && s.y[i] != null) { bd = d; bi = i; } });
    return { s, x: s.x[bi], y: s.y[bi] };
  });
  const tipX = tip.length ? tip[0].x : 0;
  const tipText = tip.map((t) => `${t.s.label} ${t.y == null ? "—" : fmtY(t.y)}`).join(", ");
  return (
    <div ref={ref} className="relative w-full select-none" data-ask-id={askId} data-ask={hx != null && tip.length ? `${ask} (Pointing at ${xLabel} ${Math.round(tipX)}: ${tipText}.)` : ask}>
      <svg width={W} height={H} className="block touch-pan-y" onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={(e) => e.pointerType === "mouse" && setHx(null)}>
        {yTicks.map((v) => <g key={v}><line x1={L} x2={L + iw} y1={Y(v)} y2={Y(v)} stroke="var(--viz-grid)" /><text x={L - 6} y={Y(v) + 3.5} fontSize={10} textAnchor="end" className="fill-muted-foreground tabular-nums">{+v.toFixed(3)}</text></g>)}
        {xTicks.map((v) => <text key={v} x={X(v)} y={H - 10} fontSize={10} textAnchor="middle" className="fill-muted-foreground tabular-nums">{fmtStep(v)}</text>)}
        {refs.map((r) => r.y <= yMax && r.y >= yMin && (
          <g key={r.label}>
            <line x1={L} x2={L + iw} y1={Y(r.y)} y2={Y(r.y)} stroke="var(--muted-foreground)" strokeDasharray="2 4" opacity={0.7} />
            <text x={L + iw - 2} y={Y(r.y) - 4} fontSize={10} textAnchor="end" className="fill-muted-foreground">{r.label}</text>
          </g>
        ))}
        {series.map((s) => <path key={s.id} d={path(s)} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dashed ? "5 4" : undefined} opacity={s.opacity ?? 1} strokeLinejoin="round" strokeLinecap="round" />)}
        {series.filter((s) => s.dots).map((s) => s.x.map((x, i) => s.y[i] != null && <circle key={s.id + i} cx={X(x)} cy={Y(s.y[i]!)} r={2.5} fill={s.color} stroke="var(--card)" strokeWidth={1.5} />))}
        {hx != null && tip.length > 0 && (
          <g pointerEvents="none">
            <line x1={X(tipX)} x2={X(tipX)} y1={T} y2={T + ih} stroke="var(--foreground)" opacity={0.35} />
            {tip.map((t) => t.y != null && <circle key={t.s.id} cx={X(t.x)} cy={Y(t.y)} r={4} fill={t.s.color} stroke="var(--card)" strokeWidth={2} />)}
          </g>
        )}
        <rect x={L} y={T} width={iw} height={ih} fill="transparent" />
      </svg>
      {hx != null && tip.length > 0 && (
        <div className="pointer-events-none absolute top-1 z-10 max-w-[70%] rounded-md border bg-popover/95 px-2 py-1 text-xs shadow-sm"
          style={X(tipX) > W / 2 ? { right: W - X(tipX) + 8 } : { left: X(tipX) + 8 }}>
          <div className="font-medium tabular-nums text-muted-foreground">{xLabel} {Math.round(tipX).toLocaleString()}</div>
          {tip.map((t) => <div key={t.s.id} className="flex items-center gap-1.5 tabular-nums"><span className="h-0.5 w-3 shrink-0 rounded" style={{ background: t.s.color }} /><span className="truncate">{t.s.label}</span><span className="ml-auto pl-2 font-medium">{t.y == null ? "—" : fmtY(t.y)}</span></div>)}
        </div>
      )}
    </div>
  );
}
