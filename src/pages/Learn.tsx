import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Check, CheckCircle2, ChevronDown, Circle, Copy, ExternalLink, FlaskConical, Play } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { CHAPTERS, TRACKS, chapterById } from "../content/curriculum";
import { CONTENT } from "../content/chapters-content";
import { nextChapter } from "@/components/app/level";
import { learner, setProgress, setScreen, useStore } from "../lib/store";
import { SeriesPlayer, loadVideos, type VideoMeta } from "./Videos";

function useVideoIndex() {
  const [vids, setVids] = useState<VideoMeta[]>([]);
  useEffect(() => { loadVideos().then(setVids).catch(() => {}); }, []);
  return vids;
}

export function Learn() {
  const progress = useStore(learner, (s) => s.progress);
  const level = useStore(learner, (s) => s.level);
  const vids = useVideoIndex();
  const done = CHAPTERS.filter((c) => progress[c.id]?.done).length;
  const next = nextChapter({ level, progress });
  useEffect(() => setScreen("Learn", "", `Chapter list: ${TRACKS.map((t) => `${t.title}: ${CHAPTERS.filter((c) => c.track === t.id).map((c) => c.title).join(", ")}`).join(" | ")}. ${done} of ${CHAPTERS.length} done; next: ${next.title}.`), [done, next.title]);
  const poster = (video?: string) => vids.find((v) => v.id === video)?.poster;

  return (
    <div className="@container mx-auto max-w-3xl space-y-8 pb-10">
      <div className="flex items-end justify-between gap-4">
        <h1 className="text-3xl font-semibold tracking-tight">Learn</h1>
        <span className="text-sm text-muted-foreground tabular-nums">{done}/{CHAPTERS.length} done</span>
      </div>

      <a href={`#/learn/${next.id}`} className="group flex items-center gap-4 rounded-xl border bg-card p-3 transition-colors hover:bg-muted/40">
        <div className="relative aspect-video w-28 shrink-0 overflow-hidden rounded-lg bg-muted">
          {poster(next.video) && <img src={poster(next.video)} alt="" className="size-full object-cover" />}
          <Play className="absolute inset-0 m-auto size-6 fill-white text-white drop-shadow" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-xs text-muted-foreground">{done ? "Continue" : "Start here"}</div>
          <div className="font-semibold">{next.title}</div>
        </div>
        <ArrowRight className="size-5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
      </a>

      {TRACKS.map((t, ti) => (
        <section key={t.id} className="space-y-1">
          <h2 className="px-1 pb-1 text-sm font-medium text-muted-foreground">{ti + 1}. {t.title}</h2>
          {CHAPTERS.filter((c) => c.track === t.id).map((c) => {
            const n = CHAPTERS.indexOf(c) + 1, p = progress[c.id];
            return (
              <a key={c.id} href={`#/learn/${c.id}`} className="flex items-center gap-3 rounded-lg px-1 py-1.5 transition-colors hover:bg-muted/60"
                data-ask-id={`chapter-${c.id}`} data-ask={`Chapter ${n}: ${c.title}`}>
                <div className="relative aspect-video w-20 shrink-0 overflow-hidden rounded-md bg-muted">
                  {poster(c.video) && <img src={poster(c.video)} alt="" loading="lazy" className="size-full object-cover" />}
                </div>
                <span className="min-w-0 flex-1"><span className="font-medium">{c.title}</span><span className="ml-2 text-xs text-muted-foreground">{c.minutes} min</span></span>
                {p?.done ? <CheckCircle2 className="size-5 shrink-0 text-emerald-500" /> : <Circle className="size-5 shrink-0 text-muted-foreground/40" />}
              </a>
            );
          })}
        </section>
      ))}
    </div>
  );
}

export function ChapterPage({ id }: { id: string }) {
  const ch = chapterById(id);
  const content = CONTENT[id];
  const vids = useVideoIndex();
  const video = vids.find((v) => v.id === ch?.video);
  const idx = ch ? CHAPTERS.indexOf(ch) : -1;
  const prev = CHAPTERS[idx - 1], next = CHAPTERS[idx + 1];
  useEffect(() => {
    if (!ch) return;
    setScreen("Learn", ch.title, [
      `Chapter ${idx + 1}/${CHAPTERS.length}: "${ch.title}".`,
      content ? `Key ideas: ${content.keyIdeas.map((k) => `${k.title}: ${k.body}`).join(" | ")}` : "",
    ].filter(Boolean).join("\n"));
  }, [ch, content, idx]);
  if (!ch) return <p className="p-10 text-center text-muted-foreground"><a className="underline" href="#/learn">Back to Learn</a></p>;

  return (
    <div className="@container mx-auto max-w-3xl space-y-6 pb-10">
      <div className="space-y-1">
        <a href="#/learn" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"><ArrowLeft className="size-3.5" />Learn</a>
        <h1 className="text-2xl font-semibold tracking-tight @2xl:text-3xl">{ch.title}</h1>
      </div>

      {video && <SeriesPlayer key={video.id} v={video} embedded label={`Chapter ${idx + 1} of ${CHAPTERS.length}`} />}

      {ch.lab && (
        <Button variant="outline" className="w-full justify-between" asChild>
          <a href={ch.lab.href}><span className="flex items-center gap-2"><FlaskConical />{ch.lab.label}</span><ArrowRight /></a>
        </Button>
      )}

      {content && (<>
        <section className="space-y-3">
          <h2 className="text-lg font-semibold tracking-tight">Key ideas</h2>
          <ul className="space-y-3">
            {content.keyIdeas.slice(0, 4).map((k, i) => (
              <li key={i} className="flex gap-3" data-ask-id={`idea-${i}`} data-ask={`Key idea "${k.title}": ${k.body}`}>
                <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-xs tabular-nums">{i + 1}</span>
                <span><span className="font-medium">{k.title}.</span> <span className="text-muted-foreground">{k.body}</span></span>
              </li>
            ))}
          </ul>
        </section>

        <Fold title="Quick check"><Quiz chapter={ch.id} questions={content.quiz} /></Fold>

        <Fold title="Go deeper">
          <div className="space-y-4">
            {content.pitfall && (
              <p className="text-sm" data-ask-id="pitfall" data-ask={`Common misconception: ${content.pitfall.myth}. Truth: ${content.pitfall.truth}`}>
                <span className="font-medium">Common mix-up: </span><span className="text-muted-foreground line-through">{content.pitfall.myth}</span> {content.pitfall.truth}
              </p>
            )}
            <pre className="overflow-x-auto rounded-lg bg-muted/50 p-3 font-mono text-[13px] leading-relaxed whitespace-pre-wrap" data-ask-id="deeper-math" data-ask={`The math for ${ch.title}: ${content.deeper.math}`}>{content.deeper.math}</pre>
            <div className="relative">
              <Button size="xs" variant="outline" className="absolute top-2 right-2" onClick={() => { navigator.clipboard.writeText(content.deeper.code); toast.success("Copied"); }}><Copy />Copy</Button>
              <pre className="max-h-[420px] overflow-auto rounded-lg border bg-[var(--viz-stage)] p-3 font-mono text-[12.5px] leading-relaxed" data-ask-id="deeper-code" data-ask={`Code example for ${ch.title}:\n${content.deeper.code}`}><code>{content.deeper.code}</code></pre>
            </div>
            {content.further?.map((f) => (
              <a key={f.url} href={f.url} target="_blank" rel="noreferrer" className="flex items-center gap-2 text-sm hover:underline"><ExternalLink className="size-3.5 shrink-0 text-muted-foreground" />{f.title}</a>
            ))}
          </div>
        </Fold>
      </>)}

      <div className="flex items-stretch gap-2 border-t pt-4">
        {prev ? <Button variant="ghost" className="h-auto flex-1 justify-start py-2 text-left whitespace-normal" asChild><a href={`#/learn/${prev.id}`}><ArrowLeft />{prev.title}</a></Button> : <span className="flex-1" />}
        {next ? <Button className="h-auto flex-1 justify-end py-2 text-right whitespace-normal" asChild onClick={() => setProgress(ch.id, { done: true })}><a href={`#/learn/${next.id}`}>{next.title}<ArrowRight /></a></Button>
          : <Button className="flex-1" asChild onClick={() => setProgress(ch.id, { done: true })}><a href="#/labs"><Check />Done. Go build something</a></Button>}
      </div>
    </div>
  );
}

function Fold({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Collapsible className="group/fold rounded-xl border">
      <CollapsibleTrigger className="flex w-full items-center justify-between p-4 text-left font-medium">
        {title}<ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[state=open]/fold:rotate-180" />
      </CollapsibleTrigger>
      <CollapsibleContent className="border-t p-4">{children}</CollapsibleContent>
    </Collapsible>
  );
}

function Quiz({ chapter, questions }: { chapter: string; questions: { q: string; options: string[]; answer: number; why: string }[] }) {
  const [picked, setPicked] = useState<(number | null)[]>(() => questions.map(() => null));
  const score = useMemo(() => picked.filter((p, i) => p === questions[i].answer).length, [picked, questions]);
  const answered = picked.filter((p) => p !== null).length;
  useEffect(() => { if (answered === questions.length) setProgress(chapter, { quiz: score }); }, [answered, score, chapter, questions.length]);
  return (
    <div className="space-y-6">
      {questions.map((q, qi) => {
        const p = picked[qi];
        return (
          <div key={qi} className="space-y-2" data-ask-id={`quiz-${qi}`} data-ask={`Quiz question: ${q.q} Options: ${q.options.join(" / ")}${p !== null ? `. The learner picked "${q.options[p]}" (${p === q.answer ? "correct" : `wrong, correct is "${q.options[q.answer]}"`})` : ""}`}>
            <p className="text-sm font-medium">{qi + 1}. {q.q}</p>
            <div className="grid gap-1.5">
              {q.options.map((o, oi) => {
                const state = p === null ? "idle" : oi === q.answer ? "right" : oi === p ? "wrong" : "dim";
                return (
                  <button key={oi} disabled={p !== null} onClick={() => setPicked(picked.map((x, i) => (i === qi ? oi : x)))}
                    className={cn("flex min-h-10 items-center gap-2.5 rounded-lg border px-3 py-2 text-left text-sm transition-colors",
                      state === "idle" && "hover:bg-muted/60", state === "right" && "border-emerald-500/60 bg-emerald-500/10",
                      state === "wrong" && "border-destructive/60 bg-destructive/10", state === "dim" && "opacity-50")}>
                    {state === "right" && <Check className="size-4 shrink-0 text-emerald-500" />}{o}
                  </button>
                );
              })}
            </div>
            {p !== null && <p className="text-sm text-muted-foreground">{q.why}</p>}
          </div>
        );
      })}
      {answered === questions.length && <p className="text-sm font-medium">{score}/{questions.length} right</p>}
    </div>
  );
}
