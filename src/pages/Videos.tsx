import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Captions, Clapperboard, ListVideo, Maximize, MessageSquare, MousePointerClick, Pause, Play, ScrollText, Volume2, VolumeX } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { ask, select, setScreen } from "../lib/store";
import { FurtherWatching } from "./FurtherWatching";

export interface VObject { id: string; label: string; box: [number, number, number, number] }
export interface VSegment { start: number; end: number; text: string; scene: string; objects: VObject[] }
export interface VideoMeta { id: string; title: string; subtitle: string; duration: number; src: string; poster: string; relates?: string; chapters: { t: number; title: string }[]; segments: VSegment[] }

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const WATCHED = "topdown-ai-watched-v1";
const readWatched = (): Record<string, number> => { try { return JSON.parse(localStorage.getItem(WATCHED) || "{}"); } catch { return {}; } };

export function Videos() {
  const [list, setList] = useState<VideoMeta[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [curId, setCurId] = useState<string | null>(null);
  useEffect(() => {
    fetch("/videos/index.json").then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then((l: VideoMeta[]) => { setList(l); setCurId((c) => c ?? (location.hash.split("/")[2] || l[0]?.id)); })
      .catch(() => setFailed(true));
  }, []);
  const cur = list?.find((v) => v.id === curId) ?? list?.[0];
  return (
    <div className="space-y-6 pb-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Videos</h1>
        <p className="mt-1 max-w-3xl text-muted-foreground">Short explainers made from the real TinyGPT in this app. Pause at any moment and click anything in the frame, or press <b>V</b> and ask. The tutor knows exactly what is being said and shown at that second.</p>
      </div>
      <Tabs defaultValue="series">
        <TabsList><TabsTrigger value="series"><Clapperboard />Our series</TabsTrigger><TabsTrigger value="further"><ListVideo />Further watching</TabsTrigger></TabsList>
        <TabsContent value="series" className="mt-4">
          {cur && list ? <SeriesPlayer key={cur.id} v={cur} list={list} onPick={(id) => { setCurId(id); history.replaceState(null, "", `#/videos/${id}`); }} />
            : <Empty className="border border-dashed"><EmptyHeader><EmptyMedia variant="icon"><Clapperboard /></EmptyMedia><EmptyTitle>{failed ? "Videos aren't published yet" : "Loading videos…"}</EmptyTitle><EmptyDescription>{failed ? "The series is still being rendered. In the meantime, try Further watching." : ""}</EmptyDescription></EmptyHeader></Empty>}
        </TabsContent>
        <TabsContent value="further" className="mt-4"><FurtherWatching /></TabsContent>
      </Tabs>
    </div>
  );
}

function SeriesPlayer({ v, list, onPick }: { v: VideoMeta; list: VideoMeta[]; onPick: (id: string) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const frame = useRef<HTMLDivElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cc, setCc] = useState(true);
  const [muted, setMuted] = useState(false);
  const [rate, setRate] = useState(1);
  const [showObjs, setShowObjs] = useState(false);
  const [watched, setWatched] = useState(readWatched);
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
  const seg = useMemo(() => v.segments.find((s) => time >= s.start && time < s.end) ?? [...v.segments].reverse().find((s) => s.start <= time) ?? v.segments[0], [v, time]);
  const segIdx = v.segments.indexOf(seg);
  const chapter = [...v.chapters].reverse().find((c) => c.t <= time + 0.01) ?? v.chapters[0];

  const seek = (t: number) => { if (video.current) { video.current.currentTime = t; setTime(t); } };
  const toggle = useCallback(() => { const el = video.current; if (!el) return; if (el.paused) el.play(); else el.pause(); }, []);

  // Tell the tutor exactly what's being said/shown right now.
  useEffect(() => {
    const prev = v.segments[segIdx - 1];
    setScreen("Videos", `${v.title} @ ${fmt(time)}`, [
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
    if (Math.floor(t) % 5 === 0) { const w = { ...readWatched(), [v.id]: Math.max(readWatched()[v.id] ?? 0, t / v.duration) }; try { localStorage.setItem(WATCHED, JSON.stringify(w)); } catch { /* */ } setWatched(w); }
  };
  const askMoment = () => {
    video.current?.pause();
    select([{ id: `video-${v.id}-t${Math.floor(time)}`, label: `"${v.title}" at ${fmt(time)}: narration "${seg?.text ?? ""}"; on screen: ${seg?.scene ?? ""}` }], "", false);
    ask.set({ open: true });
  };
  const objs = !playing ? seg?.objects ?? [] : [];

  return (
    <div className="@container"><div className="grid gap-4 @5xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="min-w-0 space-y-3">
        <div ref={frame} className="group @container/frame relative overflow-hidden rounded-xl border bg-black shadow-sm">
          <video ref={video} src={blobUrl ?? undefined} poster={v.poster} className="block aspect-video w-full" preload="metadata" playsInline muted={muted}
            onClick={toggle} onTimeUpdate={onTime} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onRateChange={() => setRate(video.current!.playbackRate)} />
          {!blobUrl && (
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1 bg-white/10"><div className="h-full bg-red-500 transition-[width]" style={{ width: `${loaded * 100}%` }} /></div>
          )}
          {/* clickable objects in the paused frame */}
          {[...objs].sort((a, b) => b.box[2] * b.box[3] - a.box[2] * a.box[3]).map((o) => (
            <div key={o.id} data-ask-id={`video-${v.id}-${o.id}`} data-ask={`In the video "${v.title}" at ${fmt(time)}: ${o.label}`}
              className={cn("absolute rounded-md border-2 border-transparent transition-colors hover:border-pin hover:bg-pin/10", showObjs && "border-dashed border-pin/70")}
              style={{ left: `${o.box[0] * 100}%`, top: `${o.box[1] * 100}%`, width: `${o.box[2] * 100}%`, height: `${o.box[3] * 100}%` }}
              title={o.label} />
          ))}
          {cc && seg && time > 0 && (
            <div className="pointer-events-none absolute inset-x-0 bottom-14 flex justify-center px-6 group-hover:bottom-16">
              <span className="max-w-[80%] rounded bg-black/75 px-[1.2cqw] py-[0.5cqw] text-center text-[clamp(11px,1.9cqw,20px)] leading-snug text-white">{seg.text}</span>
            </div>
          )}
          {!playing && (
            <div className="pointer-events-none absolute top-3 right-3 flex gap-2">
              {time === 0 ? <Badge className="pointer-events-auto cursor-pointer" onClick={toggle}><Play className="fill-current" />Play · {fmt(v.duration)}</Badge>
                : <Badge variant="secondary" className="bg-black/70 text-white"><MousePointerClick />Paused: click anything in the frame</Badge>}
            </div>
          )}
          {/* control bar */}
          <div className={cn("absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent px-3 pt-8 pb-2 text-white transition-opacity", playing && "opacity-0 group-hover:opacity-100")}>
            <div className="relative mb-1.5">
              <Slider min={0} max={v.duration} step={0.1} value={[time]} onValueChange={([t]) => seek(t)} className="[&_[data-slot=slider-range]]:bg-red-500 [&_[data-slot=slider-thumb]]:border-red-500" />
              {v.chapters.slice(1).map((c) => <span key={c.t} className="pointer-events-none absolute top-1/2 h-2 w-0.5 -translate-y-1/2 bg-black/70" style={{ left: `${(c.t / v.duration) * 100}%` }} />)}
            </div>
            <div className="flex items-center gap-1">
              <Button size="icon-sm" variant="ghost" className="text-white hover:bg-white/15 hover:text-white" onClick={toggle}>{playing ? <Pause className="fill-current" /> : <Play className="fill-current" />}</Button>
              <Button size="icon-sm" variant="ghost" className="text-white hover:bg-white/15 hover:text-white" onClick={() => setMuted(!muted)}>{muted ? <VolumeX /> : <Volume2 />}</Button>
              <span className="text-xs tabular-nums">{fmt(time)} / {fmt(v.duration)}</span>
              <span className="mx-2 hidden truncate text-xs text-white/70 sm:inline">· {chapter?.title}</span>
              <span className="flex-1" />
              <Button size="icon-sm" variant="ghost" className={cn("text-white hover:bg-white/15 hover:text-white", showObjs && "bg-white/20")} title="Show clickable objects" onClick={() => setShowObjs(!showObjs)}><MousePointerClick /></Button>
              <Button size="icon-sm" variant="ghost" className={cn("text-white hover:bg-white/15 hover:text-white", cc && "bg-white/20")} title="Captions" onClick={() => setCc(!cc)}><Captions /></Button>
              <Select value={String(rate)} onValueChange={(r) => { if (video.current) video.current.playbackRate = +r; }}>
                <SelectTrigger size="sm" className="h-7 w-[64px] border-white/20 bg-transparent text-xs text-white dark:bg-transparent"><SelectValue /></SelectTrigger>
                <SelectContent>{[0.75, 1, 1.25, 1.5, 2].map((r) => <SelectItem key={r} value={String(r)}>{r}×</SelectItem>)}</SelectContent>
              </Select>
              <Button size="icon-sm" variant="ghost" className="text-white hover:bg-white/15 hover:text-white" onClick={() => frame.current?.requestFullscreen?.()}><Maximize /></Button>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-xl font-semibold tracking-tight">{v.title}</h2>
            <p className="text-sm text-muted-foreground">{v.subtitle}</p>
          </div>
          <div className="flex gap-2">
            <Button onClick={askMoment}><MessageSquare />Ask about {fmt(time)}</Button>
            {v.relates && <Button variant="outline" asChild><a href={v.relates === "training" ? "#/training" : "#/xray"}>Try it live</a></Button>}
          </div>
        </div>
      </div>

      <Card className="min-h-0 gap-0 py-0 @5xl:max-h-[calc(100svh-9rem)]">
        <Tabs defaultValue="chapters" className="flex min-h-0 flex-1 flex-col gap-0">
          <div className="border-b p-2"><TabsList className="w-full"><TabsTrigger value="chapters"><ListVideo />Chapters</TabsTrigger><TabsTrigger value="transcript"><ScrollText />Transcript</TabsTrigger></TabsList></div>
          <TabsContent value="chapters" className="min-h-0 flex-1">
            <ScrollArea className="h-full max-h-[60vh] @5xl:max-h-none">
              <div className="p-2">
                {v.chapters.map((c, i) => (
                  <button key={c.t} onClick={() => seek(c.t + 0.01)} className={cn("flex w-full items-center gap-3 rounded-md px-2 py-2 text-left text-sm hover:bg-muted", chapter === c && "bg-muted font-medium")}>
                    <span className="w-5 text-xs text-muted-foreground tabular-nums">{i + 1}</span><span className="flex-1">{c.title}</span><span className="text-xs text-muted-foreground tabular-nums">{fmt(c.t)}</span>
                  </button>
                ))}
              </div>
              <div className="border-t p-2">
                <div className="px-2 pb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">Up next in the series</div>
                {list.map((x, i) => (
                  <button key={x.id} onClick={() => onPick(x.id)} className={cn("flex w-full gap-3 rounded-lg p-1.5 text-left hover:bg-muted", x.id === v.id && "bg-muted")}>
                    <div className="relative w-28 shrink-0 overflow-hidden rounded-md">
                      <img src={x.poster} alt="" className="aspect-video w-full object-cover" loading="lazy" />
                      <span className="absolute right-1 bottom-1 rounded bg-black/80 px-1 text-[10px] text-white tabular-nums">{fmt(x.duration)}</span>
                      {watched[x.id] ? <span className="absolute inset-x-0 bottom-0 h-0.5 bg-red-500" style={{ width: `${Math.min(100, watched[x.id] * 100)}%` }} /> : null}
                    </div>
                    <span className="min-w-0 text-sm leading-snug"><span className="line-clamp-2 font-medium">{i + 1}. {x.title}</span><span className="line-clamp-1 text-xs text-muted-foreground">{x.subtitle}</span></span>
                  </button>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
          <TabsContent value="transcript" className="min-h-0 flex-1">
            <Transcript v={v} segIdx={segIdx} seek={seek} />
          </TabsContent>
        </Tabs>
      </Card>
    </div></div>
  );
}

function Transcript({ v, segIdx, seek }: { v: VideoMeta; segIdx: number; seek: (t: number) => void }) {
  const cur = useRef<HTMLButtonElement>(null);
  useEffect(() => { cur.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }, [segIdx]);
  return (
    <ScrollArea className="h-full max-h-[60vh] xl:max-h-[calc(100svh-13rem)]">
      <div className="space-y-0.5 p-2">
        {v.segments.map((s, i) => (
          <button key={i} ref={i === segIdx ? cur : undefined} onClick={() => seek(s.start + 0.01)}
            className={cn("flex w-full gap-3 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted", i === segIdx ? "bg-muted text-foreground" : "text-muted-foreground")}>
            <span className="w-9 shrink-0 pt-0.5 text-xs tabular-nums">{fmt(s.start)}</span><span>{s.text}</span>
          </button>
        ))}
      </div>
    </ScrollArea>
  );
}
