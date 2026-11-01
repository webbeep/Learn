import { useEffect, useRef, useState } from "react";
import { MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { ask, select, setScreen } from "../lib/store";

const VIDEOS = [
  { id: "wjZofJX0v4M", title: "Transformers, the tech behind LLMs", who: "3Blue1Brown", level: "Big picture", note: "The best visual tour of the whole pipeline you just X-rayed." },
  { id: "eMlx5fFNoYc", title: "Attention in transformers, step-by-step", who: "3Blue1Brown", level: "Big picture", note: "Pairs with scene 3 (Attention)." },
  { id: "9-Jl0dxWQs8", title: "How might LLMs store facts", who: "3Blue1Brown", level: "Big picture", note: "Pairs with scene 4 (MLP neurons)." },
  { id: "zjkBMFhNj_g", title: "Intro to Large Language Models", who: "Andrej Karpathy", level: "Big picture", note: "How real LLMs are trained and used, end to end." },
  { id: "zduSFxRajkE", title: "Let's build the GPT Tokenizer", who: "Andrej Karpathy", level: "Build it", note: "Pairs with scene 1 — builds BPE like ours." },
  { id: "kCc8FmEb1nY", title: "Let's build GPT: from scratch, in code", who: "Andrej Karpathy", level: "Build it", note: "Basically TinyGPT, written live. Your on-ramp to training from scratch." },
  { id: "aircAruvnKk", title: "But what is a neural network?", who: "3Blue1Brown", level: "Dive down", note: "Dive down when you want the foundations." },
  { id: "IHZwWFHWa-w", title: "Gradient descent, how neural networks learn", who: "3Blue1Brown", level: "Dive down", note: "Pairs with the Training peek." },
  { id: "Ilg3gGewQ5U", title: "Backpropagation, intuitively", who: "3Blue1Brown", level: "Dive down", note: "How every weight knows which way to move." },
  { id: "VMj-3S1tku0", title: "Building micrograd (backprop from scratch)", who: "Andrej Karpathy", level: "Dive down", note: "Build the engine that trains every model." },
];

/* eslint-disable @typescript-eslint/no-explicit-any */
declare global { interface Window { YT?: any; onYouTubeIframeAPIReady?: () => void } }
let ytReady: Promise<void> | null = null;
function loadYT() {
  ytReady ??= new Promise((res) => {
    if (window.YT?.Player) return res();
    window.onYouTubeIframeAPIReady = () => res();
    const s = document.createElement("script"); s.src = "https://www.youtube.com/iframe_api"; document.head.appendChild(s);
  });
  return ytReady;
}
const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function FurtherWatching() {
  const [cur, setCur] = useState(VIDEOS[0]);
  const player = useRef<any>(null);
  const holder = useRef<HTMLDivElement>(null);
  const [time, setTime] = useState(0);

  useEffect(() => {
    let alive = true;
    loadYT().then(() => {
      if (!alive || !holder.current) return;
      player.current?.destroy?.();
      const div = document.createElement("div"); holder.current.innerHTML = ""; holder.current.appendChild(div);
      player.current = new window.YT.Player(div, { videoId: cur.id, playerVars: { rel: 0, modestbranding: 1 },
        events: { onStateChange: (e: any) => { if (e.data === 2) pauseAsk(); } } });
    });
    return () => { alive = false; };
  }, [cur]); // eslint-disable-line

  useEffect(() => {
    const iv = setInterval(() => { const t = player.current?.getCurrentTime?.(); if (typeof t === "number") setTime(t); }, 1000);
    return () => clearInterval(iv);
  }, []);
  useEffect(() => {
    setScreen("Videos", cur.title, `Video: "${cur.title}" by ${cur.who}. Learner is at ${fmt(time)}. (You cannot see the video frame; use your knowledge of this well-known video and ask the learner to describe the frame if needed.) Note: ${cur.note}`);
  }, [cur, Math.floor(time / 5)]); // eslint-disable-line

  function pauseAsk() {
    const t = player.current?.getCurrentTime?.() ?? 0;
    select([{ id: `video-${cur.id}-${Math.floor(t)}`, label: `"${cur.title}" (${cur.who}) at ${fmt(t)}` }], "", false);
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
      <div className="space-y-3">
        <div className="aspect-video overflow-hidden rounded-xl border bg-black [&>div]:size-full [&_iframe]:size-full" ref={holder} />
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => { player.current?.pauseVideo?.(); pauseAsk(); ask.set({ open: true }); }}><MessageSquare />Pause & ask about {fmt(time)}</Button>
          <span className="text-sm text-muted-foreground">{cur.note}</span>
        </div>
        <p className="text-xs text-muted-foreground">The tutor knows these well-known videos and the timestamp, but it can't see YouTube's frames. Our own series can be clicked when paused.</p>
      </div>
      <ScrollArea className="h-[70vh] rounded-xl border">
        <div className="space-y-4 p-3">
          {["Big picture", "Build it", "Dive down"].map((lvl) => (
            <div key={lvl} className="space-y-1.5">
              <div className="px-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">{lvl}</div>
              {VIDEOS.filter((v) => v.level === lvl).map((v) => (
                <button key={v.id} onClick={() => setCur(v)} className={cn("flex w-full gap-3 rounded-lg p-1.5 text-left transition-colors hover:bg-muted", v.id === cur.id && "bg-muted")}>
                  <img src={`https://i.ytimg.com/vi/${v.id}/mqdefault.jpg`} alt="" loading="lazy" className="aspect-video w-28 shrink-0 rounded-md object-cover" />
                  <span className="min-w-0 text-sm leading-snug"><span className="line-clamp-2 font-medium">{v.title}</span><span className="text-xs text-muted-foreground">{v.who}</span></span>
                </button>
              ))}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
