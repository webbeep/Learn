import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Captions, Check, ChevronDown, Maximize, MessageSquare, MousePointerClick, Pause, Play, ScrollText, SkipForward, Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { ask, select, setProgress, setScreen } from "../lib/store";
import { FurtherWatching } from "./FurtherWatching";
import { CHAPTERS, TRACKS } from "../content/curriculum";

export interface VObject { id: string; label: string; box: [number, number, number, number] }
export interface VSegment { start: number; end: number; text: string; scene: string; objects: VObject[] }
export interface VideoMeta { id: string; title: string; subtitle: string; duration: number; src: string; poster: string; relates?: string; track?: string; order?: number; chapters: { t: number; title: string }[]; segments: VSegment[] }

let indexPromise: Promise<VideoMeta[]> | null = null;
export function loadVideos() {
  indexPromise ??= fetch("/videos/index.json").then((r) => { if (!r.ok) throw new Error("no index"); return r.json(); });
  return indexPromise;
}
const LAB: Record<string, string> = { xray: "#/xray", training: "#/training", train: "#/train", backprop: "#/backprop", lab: "#/lab", scaling: "#/scaling", invent: "#/invent" };
const chapterFor = (videoId: string) => CHAPTERS.find((c) => c.video === videoId);
/** Short, friendly name for a video (the chapter title when there is one). */
export const videoName = (v: VideoMeta) => chapterFor(v.id)?.title ?? v.title;

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const WATCHED = "topdown-ai-watched-v1";
const readWatched = (): Record<string, number> => { try { return JSON.parse(localStorage.getItem(WATCHED) || "{}"); } catch { return {}; } };
const AUTONEXT = "topdown-ai-autonext";

/** Watch: the whole series as one continuous show. */
export function Watch() {
  const [list, setList] = useState<VideoMeta[] | null>(null);
  const [curId, setCurId] = useState<string | null>(null);
  const [autoNext, setAutoNext] = useState(() => { try { return localStorage.getItem(AUTONEXT) !== "0"; } catch { return true; } });
  const [autoPlay, setAutoPlay] = useState(false);
  const [watched, setWatched] = useState(readWatched);
  useEffect(() => {
    loadVideos().then((l) => {
      setList(l);
      const fromHash = location.hash.split("/")[2];
      const w = readWatched();
      setCurId((c) => c ?? (l.find((v) => v.id === fromHash) ?? l.find((v) => (w[v.id] ?? 0) < 0.9) ?? l[0])?.id ?? null);
    }).catch(() => setList([]));
  }, []);
  const idx = list?.findIndex((v) => v.id === curId) ?? -1;
  const cur = list && idx >= 0 ? list[idx] : undefined;
  const pick = (id: string, play = true) => { setCurId(id); setAutoPlay(play); history.replaceState(null, "", `#/watch/${id}`); };
  const next = list?.[idx + 1];

  if (!list) return <div className="flex items-center justify-center gap-2 p-16 text-muted-foreground"><Spinner />Loading…</div>;
  if (!cur) return <p className="p-10 text-center text-muted-foreground">No videos yet.</p>;
  const total = list.reduce((a, v) => a + v.duration, 0);

  return (
    <div className="@container pb-10">
      <div className="grid gap-6 @5xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0 space-y-4">
          <SeriesPlayer key={cur.id} v={cur} autoPlay={autoPlay}
            label={`${idx + 1} of ${list.length}`}
            onProgress={() => setWatched(readWatched())}
            onEnded={() => { if (autoNext && next) pick(next.id); }} />
          {next && (
            <button onClick={() => pick(next.id)} className="flex w-full items-center gap-3 rounded-xl border p-2 text-left transition-colors hover:bg-muted/50 @5xl:hidden">
              <img src={next.poster} alt="" className="aspect-video w-24 rounded-md object-cover" />
              <span className="min-w-0 flex-1"><span className="block text-xs text-muted-foreground">Up next</span><span className="block truncate font-medium">{videoName(next)}</span></span>
              <SkipForward className="size-4 text-muted-foreground" />
            </button>
          )}
        </div>

        <div className="@5xl:sticky @5xl:top-20 @5xl:max-h-[calc(100svh-6rem)] @5xl:overflow-y-auto">
          <div className="mb-2 flex items-center justify-between gap-2 px-1">
            <span className="text-sm font-medium">{list.length} videos · {Math.round(total / 60)} min</span>
            <label className="flex items-center gap-2 text-xs text-muted-foreground">Autoplay
              <Switch checked={autoNext} onCheckedChange={(v) => { setAutoNext(v); try { localStorage.setItem(AUTONEXT, v ? "1" : "0"); } catch { /* */ } }} />
            </label>
          </div>
          {TRACKS.map((t) => {
            const vids = list.filter((v) => v.track === t.id);
            if (!vids.length) return null;
            return (
              <div key={t.id} className="mb-3">
                <div className="px-1 pb-1 text-xs font-medium text-muted-foreground">{t.title}</div>
                {vids.map((v) => {
                  const n = list.indexOf(v) + 1, done = (watched[v.id] ?? 0) > 0.9, on = v.id === cur.id;
                  return (
                    <button key={v.id} onClick={() => pick(v.id)} className={cn("flex w-full items-center gap-3 rounded-lg px-1 py-1.5 text-left transition-colors hover:bg-muted", on && "bg-muted")}>
                      <div className="relative w-20 shrink-0 overflow-hidden rounded-md">
                        <img src={v.poster} alt="" loading="lazy" className="aspect-video w-full object-cover" />
                        {(watched[v.id] ?? 0) > 0.02 && <span className="absolute bottom-0 left-0 h-0.5 bg-red-500" style={{ width: `${Math.min(100, (watched[v.id] ?? 0) * 100)}%` }} />}
                      </div>
                      <span className="min-w-0 flex-1 text-sm leading-tight"><span className={cn("line-clamp-2", on && "font-medium")}>{n}. {videoName(v)}</span><span className="text-xs text-muted-foreground tabular-nums">{fmt(v.duration)}</span></span>
                      {done && <Check className="size-4 shrink-0 text-emerald-500" />}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      <Collapsible className="group/fw mt-8 border-t pt-4">
        <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
          More videos by 3Blue1Brown and Andrej Karpathy <ChevronDown className="size-4 transition-transform group-data-[state=open]/fw:rotate-180" />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-4"><FurtherWatching /></CollapsibleContent>
      </Collapsible>
    </div>
  );
}
export const Videos = Watch;

export function SeriesPlayer({ v, embedded, autoPlay, onEnded, onProgress, label }: {
  v: VideoMeta; embedded?: boolean; autoPlay?: boolean; onEnded?: () => void; onProgress?: () => void; label?: string;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const frame = useRef<HTMLDivElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cc, setCc] = useState(true);
  const [muted, setMuted] = useState(false);
  const [rate, setRate] = useState(1);
  const [showObjs, setShowObjs] = useState(false);
  const [showText, setShowText] = useState(false);
  // Download the whole (small) file and play it from memory: seeking then works on any host,
  // even ones that don't honor HTTP range requests.
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(0);
  useEffect(() => {
    let url = "", alive = true;
    (async () => {
      try {
        const res = await fetch(v.src); const total = +(res.headers.get("content-length") || 0);
        const reader = res.body!.getReader(); const chunks: Uint8Array[] = []; let got = 0;
        for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (alive && total) setLoaded(got / total); }
        url = URL.createObjectURL(new Blob(chunks as BlobPart[], { type: "video/mp4" }));
        if (alive) setBlobUrl(url);
      } catch { if (alive) setBlobUrl(v.src); }
    })();
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [v.src]);
  useEffect(() => { if (blobUrl && autoPlay) video.current?.play().catch(() => { /* needs a tap first */ }); }, [blobUrl, autoPlay]);

  const seg = useMemo(() => v.segments.find((s) => time >= s.start && time < s.end) ?? [...v.segments].reverse().find((s) => s.start <= time) ?? v.segments[0], [v, time]);
  const segIdx = v.segments.indexOf(seg);
  const chapter = [...v.chapters].reverse().find((c) => c.t <= time + 0.01) ?? v.chapters[0];
  const name = videoName(v);

  const seek = (t: number) => { if (video.current) { video.current.currentTime = t; setTime(t); } };
  const toggle = useCallback(() => { const el = video.current; if (!el) return; if (el.paused) el.play(); else el.pause(); }, []);

  // Tell the tutor exactly what's being said/shown right now.
  useEffect(() => {
    const prev = v.segments[segIdx - 1];
    setScreen("Watch", `${name} @ ${fmt(time)}`, [
      `Video "${v.title}" (${v.subtitle}), generated from the real TinyGPT in this app. Learner is at ${fmt(time)} of ${fmt(v.duration)}${playing ? " (playing)" : " (paused)"}. Chapter: ${chapter?.title ?? "-"}.`,
      prev ? `Narration just before: "${prev.text}"` : "",
      seg ? `Narration right now: "${seg.text}"` : "",
      seg ? `On screen right now: ${seg.scene}` : "",
      seg?.objects.length ? `Objects in the frame: ${seg.objects.map((o) => o.label).join("; ")}` : "",
    ].filter(Boolean).join("\n"));
  }, [v, segIdx, Math.floor(time), playing]); // eslint-disable-line

  // clicking / selecting anything pauses the video
  useEffect(() => ask.subscribe(() => { if (ask.get().selection.length && video.current && !video.current.paused) video.current.pause(); }), []);
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, [role=slider], [role=combobox]")) return;
      if (e.key === " ") { e.preventDefault(); toggle(); }
      if (e.key === "ArrowRight") seek(Math.min(v.duration, (video.current?.currentTime ?? 0) + 5));
      if (e.key === "ArrowLeft") seek(Math.max(0, (video.current?.currentTime ?? 0) - 5));
    };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  }, [toggle, v.duration]);

  const onTime = () => {
    const t = video.current!.currentTime; setTime(t);
    if (Math.floor(t) % 5 === 0 || t >= v.duration - 1) {
      const w = { ...readWatched(), [v.id]: Math.max(readWatched()[v.id] ?? 0, t / v.duration) }; try { localStorage.setItem(WATCHED, JSON.stringify(w)); } catch { /* */ }
      const ch = chapterFor(v.id); if (ch) setProgress(ch.id, { watched: t / v.duration });
      onProgress?.();
    }
  };
  const askMoment = () => {
    video.current?.pause();
    select([{ id: `video-${v.id}-t${Math.floor(time)}`, label: `"${name}" at ${fmt(time)}: narration "${seg?.text ?? ""}"; on screen: ${seg?.scene ?? ""}` }], "", false);
    ask.set({ open: true });
  };
  const objs = !playing ? seg?.objects ?? [] : [];
  const wbtn = "text-white hover:bg-white/15 hover:text-white";

  return (
    <div className="space-y-3">
      <div ref={frame} className="group @container/frame relative overflow-hidden rounded-xl bg-black shadow-sm">
        <video ref={video} src={blobUrl ?? undefined} poster={v.poster} className="block aspect-video w-full" preload="metadata" playsInline muted={muted}
          onClick={toggle} onTimeUpdate={onTime} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
          onEnded={() => { setPlaying(false); onEnded?.(); }} onRateChange={() => setRate(video.current!.playbackRate)} />
        {!blobUrl && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1 bg-white/10"><div className="h-full bg-red-500 transition-[width]" style={{ width: `${loaded * 100}%` }} /></div>
        )}
        {/* clickable objects in the paused frame */}
        {[...objs].sort((a, b) => b.box[2] * b.box[3] - a.box[2] * a.box[3]).map((o) => (
          <div key={o.id} data-ask-id={`video-${v.id}-${o.id}`} data-ask={`In the video "${name}" at ${fmt(time)}: ${o.label}`}
            className={cn("absolute rounded-md border-2 border-transparent transition-colors hover:border-pin hover:bg-pin/10", showObjs && "border-dashed border-pin/70")}
            style={{ left: `${o.box[0] * 100}%`, top: `${o.box[1] * 100}%`, width: `${o.box[2] * 100}%`, height: `${o.box[3] * 100}%` }}
            title={o.label} />
        ))}
        {cc && seg && time > 0 && (
          <div className="pointer-events-none absolute inset-x-0 bottom-14 flex justify-center px-6 group-hover:bottom-16">
            <span className="max-w-[80%] rounded bg-black/75 px-[1.2cqw] py-[0.5cqw] text-center text-[clamp(11px,1.9cqw,20px)] leading-snug text-white">{seg.text}</span>
          </div>
        )}
        {!playing && time > 0 && (
          <div className="pointer-events-none absolute top-3 right-3 hidden rounded-full bg-black/70 px-3 py-1 text-xs text-white @md/frame:block">Tap anything to ask about it</div>
        )}
        {!playing && time === 0 && (
          <button onClick={toggle} aria-label="Play" className="absolute inset-0 m-auto flex size-16 items-center justify-center rounded-full bg-black/65 text-white shadow-xl ring-1 ring-white/20 backdrop-blur transition-transform hover:scale-105 @md/frame:size-20">
            <Play className="size-7 translate-x-0.5 fill-current @md/frame:size-9" />
          </button>
        )}
        {/* control bar */}
        <div className={cn("absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent px-3 pt-8 pb-2 text-white transition-opacity", playing && "opacity-0 group-hover:opacity-100")}>
          <div className="relative mb-1.5">
            <Slider min={0} max={v.duration} step={0.1} value={[time]} onValueChange={([t]) => seek(t)} className="[&_[data-slot=slider-range]]:bg-red-500 [&_[data-slot=slider-thumb]]:border-red-500" />
            {v.chapters.slice(1).map((c) => <span key={c.t} className="pointer-events-none absolute top-1/2 h-2 w-0.5 -translate-y-1/2 bg-black/70" style={{ left: `${(c.t / v.duration) * 100}%` }} />)}
          </div>
          <div className="flex items-center gap-1">
            <Button size="icon-sm" variant="ghost" className={wbtn} onClick={toggle} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause className="fill-current" /> : <Play className="fill-current" />}</Button>
            <Button size="icon-sm" variant="ghost" className={wbtn} onClick={() => setMuted(!muted)} aria-label="Mute">{muted ? <VolumeX /> : <Volume2 />}</Button>
            <span className="text-xs tabular-nums">{fmt(time)} / {fmt(v.duration)}</span>
            <span className="mx-2 hidden truncate text-xs text-white/70 @lg/frame:inline">{chapter?.title}</span>
            <span className="flex-1" />
            <Button size="icon-sm" variant="ghost" className={cn("hidden @md/frame:inline-flex", wbtn, showObjs && "bg-white/20")} title="Show what you can tap" onClick={() => setShowObjs(!showObjs)}><MousePointerClick /></Button>
            <Button size="icon-sm" variant="ghost" className={cn(wbtn, cc && "bg-white/20")} title="Captions" onClick={() => setCc(!cc)}><Captions /></Button>
            <Select value={String(rate)} onValueChange={(r) => { if (video.current) video.current.playbackRate = +r; }}>
              <SelectTrigger size="sm" className="hidden h-7 w-[64px] border-white/20 bg-transparent text-xs text-white @sm/frame:flex dark:bg-transparent"><SelectValue /></SelectTrigger>
              <SelectContent>{[0.75, 1, 1.25, 1.5, 2].map((r) => <SelectItem key={r} value={String(r)}>{r}×</SelectItem>)}</SelectContent>
            </Select>
            <Button size="icon-sm" variant="ghost" className={wbtn} onClick={() => frame.current?.requestFullscreen?.()} aria-label="Full screen"><Maximize /></Button>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          {label && <div className="text-xs text-muted-foreground">{label}</div>}
          {!embedded && <h1 className="text-xl font-semibold tracking-tight">{name}</h1>}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" onClick={askMoment}><MessageSquare />Ask</Button>
          {v.relates && LAB[v.relates] && <Button size="sm" variant="outline" asChild><a href={LAB[v.relates]}>Try it</a></Button>}
          <Button size="sm" variant="ghost" onClick={() => setShowText(!showText)}><ScrollText />Transcript</Button>
        </div>
      </div>
      {showText && <Transcript v={v} segIdx={segIdx} seek={seek} />}
    </div>
  );
}

function Transcript({ v, segIdx, seek }: { v: VideoMeta; segIdx: number; seek: (t: number) => void }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => { // scroll inside the box only, never the page
    const el = box.current?.querySelector<HTMLElement>(`[data-i="${segIdx}"]`);
    if (el && box.current) box.current.scrollTo({ top: el.offsetTop - box.current.clientHeight / 3, behavior: "smooth" });
  }, [segIdx]);
  return (
    <div ref={box} className="relative max-h-64 space-y-0.5 overflow-y-auto rounded-xl border p-2">
      {v.segments.map((s, i) => (
        <button key={i} data-i={i} onClick={() => seek(s.start + 0.01)}
          className={cn("flex w-full gap-3 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted", i === segIdx ? "bg-muted text-foreground" : "text-muted-foreground")}>
          <span className="w-9 shrink-0 pt-0.5 text-xs tabular-nums">{fmt(s.start)}</span><span>{s.text}</span>
        </button>
      ))}
    </div>
  );
}
