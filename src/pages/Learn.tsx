import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, BookOpen, Check, CheckCircle2, ChevronDown, Circle, CircleDot, Clock, Copy, ExternalLink, FlaskConical, GraduationCap, Lightbulb, PlayCircle, TriangleAlert } from "lucide-react";
import { toast } from "sonner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { CHAPTERS, TRACKS, chapterById, chapterContent, type Chapter, type Level } from "../content/curriculum";
import { learner, setProgress, setScreen, useStore, type Learner } from "../lib/store";
import { SeriesPlayer, loadVideos, type VideoMeta } from "./Videos";

export const LEVEL_LABEL: Record<Level, string> = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" };
const LEVEL_RANK: Record<Level, number> = { beginner: 0, intermediate: 1, advanced: 2 };
const levelBadge = (l: Level) => (
  <Badge variant="outline" className={cn("font-normal",
    l === "beginner" && "border-emerald-500/40 text-emerald-700 dark:text-emerald-400",
    l === "intermediate" && "border-sky-500/40 text-sky-700 dark:text-sky-400",
    l === "advanced" && "border-violet-500/40 text-violet-700 dark:text-violet-400")}>{LEVEL_LABEL[l]}</Badge>
);

function useVideoIndex() {
  const [vids, setVids] = useState<VideoMeta[]>([]);
  useEffect(() => { loadVideos().then(setVids).catch(() => {}); }, []);
  return vids;
}

/** The next chapter to do, honoring the learner's level (advanced folks can skip beginner chapters). */
export function nextChapter(l: Pick<Learner, "level" | "progress">): Chapter {
  const rank = LEVEL_RANK[l.level ?? "beginner"];
  return CHAPTERS.find((c) => !l.progress[c.id]?.done && (rank < 2 || LEVEL_RANK[c.level] > 0)) ?? CHAPTERS.find((c) => !l.progress[c.id]?.done) ?? CHAPTERS[0];
}

function StatusIcon({ id }: { id: string }) {
  const p = useStore(learner, (s) => s.progress[id]);
  if (p?.done) return <CheckCircle2 className="size-5 text-emerald-500" />;
  if (p && ((p.watched ?? 0) > 0.05 || p.quiz !== undefined)) return <CircleDot className="size-5 text-pin" />;
  return <Circle className="size-5 text-muted-foreground/50" />;
}

export function LevelSelect({ className }: { className?: string }) {
  const level = useStore(learner, (s) => s.level);
  return (
    <Select value={level ?? "beginner"} onValueChange={(v) => setLevel(v as Level)}>
      <SelectTrigger size="sm" className={cn("w-[150px]", className)}><GraduationCap /><SelectValue /></SelectTrigger>
      <SelectContent>{(Object.keys(LEVEL_LABEL) as Level[]).map((k) => <SelectItem key={k} value={k}>{LEVEL_LABEL[k]}</SelectItem>)}</SelectContent>
    </Select>
  );
}
export function setLevel(level: Level) {
  learner.set({ level, depth: level === "beginner" ? "intuition" : level === "advanced" ? "math" : "balanced" });
}

export function Learn() {
  const l = useStore(learner, (s) => s);
  const vids = useVideoIndex();
  const done = CHAPTERS.filter((c) => l.progress[c.id]?.done).length;
  const next = nextChapter(l);
  useEffect(() => setScreen("Curriculum", "", `Curriculum with ${CHAPTERS.length} chapters in ${TRACKS.length} tracks: ${TRACKS.map((t) => `${t.title} (${CHAPTERS.filter((c) => c.track === t.id).map((c) => c.title).join("; ")})`).join(" | ")}. Learner level ${l.level ?? "not set"}, ${done} chapters done, suggested next: ${next.title}.`), [done, l.level, next.title]);
  const poster = (c: Chapter) => vids.find((v) => v.id === c.video)?.poster;

  return (
    <div className="@container space-y-8 pb-10">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Curriculum</h1>
          <p className="mt-1 max-w-2xl text-muted-foreground">{CHAPTERS.length} short chapters in 4 tracks. Each one has a video, a lab to try it in, key ideas, a quick check, and deeper math and code if you want them.</p>
        </div>
        <LevelSelect />
      </div>

      <Card className="overflow-hidden py-0">
        <div className="grid @2xl:grid-cols-[1fr_280px]">
          <a href={`#/learn/${next.id}`} className="group flex items-center gap-4 p-4 transition-colors hover:bg-muted/40 @md:p-5">
            <div className="relative hidden aspect-video w-36 shrink-0 overflow-hidden rounded-lg border bg-muted @md:block">
              {poster(next) ? <img src={poster(next)} alt="" className="size-full object-cover" /> : <PlayCircle className="absolute inset-0 m-auto size-8 text-muted-foreground" />}
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-medium text-pin">{done ? "Continue" : "Start here"}</div>
              <div className="text-lg font-semibold leading-snug">{next.title}</div>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">{levelBadge(next.level)}<span className="flex items-center gap-1"><Clock className="size-3" />{next.minutes} min</span></div>
            </div>
            <ArrowRight className="size-5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
          </a>
          <div className="space-y-2 border-t bg-muted/30 p-4 @2xl:border-t-0 @2xl:border-l @md:p-5">
            <div className="flex justify-between text-sm"><span className="font-medium">Your progress</span><span className="text-muted-foreground tabular-nums">{done}/{CHAPTERS.length}</span></div>
            <Progress value={(done / CHAPTERS.length) * 100} />
            <div className="text-xs text-muted-foreground">A chapter counts as done when you watch most of its video or get 2 of 3 checks right.</div>
          </div>
        </div>
      </Card>

      {TRACKS.map((t, ti) => {
        const chs = CHAPTERS.filter((c) => c.track === t.id);
        const tDone = chs.filter((c) => l.progress[c.id]?.done).length;
        return (
          <section key={t.id} className="space-y-3">
            <div className="flex items-end justify-between gap-3">
              <div>
                <div className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Track {ti + 1}</div>
                <h2 className="text-xl font-semibold tracking-tight">{t.title}</h2>
                <p className="text-sm text-muted-foreground">{t.blurb}</p>
              </div>
              <span className="shrink-0 text-xs text-muted-foreground tabular-nums">{tDone}/{chs.length} done</span>
            </div>
            <div className="grid gap-2 @3xl:grid-cols-2">
              {chs.map((c) => {
                const n = CHAPTERS.indexOf(c) + 1;
                const p = poster(c);
                return (
                  <a key={c.id} href={`#/learn/${c.id}`} className="group flex items-center gap-3 rounded-xl border bg-card p-2.5 transition-colors hover:bg-muted/50"
                    data-ask-id={`chapter-${c.id}`} data-ask={`Chapter ${n}: ${c.title} (${LEVEL_LABEL[c.level]}, ${c.minutes} min)`}>
                    <div className="relative aspect-video w-28 shrink-0 overflow-hidden rounded-lg border bg-muted">
                      {p ? <img src={p} alt="" loading="lazy" className="size-full object-cover" /> : <BookOpen className="absolute inset-0 m-auto size-6 text-muted-foreground" />}
                      <span className="absolute top-1 left-1 rounded bg-black/70 px-1.5 text-[10px] font-medium text-white tabular-nums">{n}</span>
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="line-clamp-2 text-sm font-medium leading-snug">{c.title}</div>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">{levelBadge(c.level)}<span>{c.minutes} min</span>{c.lab && <span className="hidden items-center gap-1 @md:flex"><FlaskConical className="size-3" />lab</span>}</div>
                    </div>
                    <StatusIcon id={c.id} />
                  </a>
                );
              })}
            </div>
          </section>
        );
      })}
    </div>
  );
}

export function ChapterPage({ id }: { id: string }) {
  const ch = chapterById(id);
  const content = chapterContent(id);
  const vids = useVideoIndex();
  const level = useStore(learner, (s) => s.level);
  const video = vids.find((v) => v.id === ch?.video);
  const idx = ch ? CHAPTERS.indexOf(ch) : -1;
  const prev = CHAPTERS[idx - 1], next = CHAPTERS[idx + 1];
  const track = TRACKS.find((t) => t.id === ch?.track);
  useEffect(() => {
    if (!ch) return;
    setScreen("Curriculum", ch.title, [
      `Chapter ${idx + 1}/${CHAPTERS.length} "${ch.title}" (${ch.level}, track: ${track?.title}).`,
      content ? `Promise: ${content.promise}` : "",
      content ? `Key ideas: ${content.keyIdeas.map((k) => `${k.title}: ${k.body}`).join(" | ")}` : "",
      content?.pitfall ? `Misconception: ${content.pitfall.myth} → ${content.pitfall.truth}` : "",
    ].filter(Boolean).join("\n"));
  }, [ch, content, idx, track]);
  if (!ch) return <Empty><EmptyHeader><EmptyTitle>Chapter not found</EmptyTitle><EmptyDescription><a className="underline" href="#/learn">Back to the curriculum</a></EmptyDescription></EmptyHeader></Empty>;

  return (
    <div className="@container mx-auto max-w-5xl space-y-6 pb-10">
      <div className="space-y-2">
        <a href="#/learn" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"><ArrowLeft className="size-3.5" />{track?.title}</a>
        <h1 className="text-2xl font-semibold tracking-tight text-balance @2xl:text-3xl">{ch.title}</h1>
        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          <span>Chapter {idx + 1} of {CHAPTERS.length}</span>{levelBadge(ch.level)}<span className="flex items-center gap-1"><Clock className="size-3.5" />{ch.minutes} min</span><StatusIcon id={ch.id} />
        </div>
        {content && <p className="max-w-3xl text-pretty text-muted-foreground @2xl:text-lg">{content.promise}</p>}
      </div>

      {ch.video && (video ? <SeriesPlayer key={video.id} v={video} embedded /> : (
        <Empty className="border border-dashed"><EmptyHeader><EmptyMedia variant="icon"><PlayCircle /></EmptyMedia><EmptyTitle>Video on its way</EmptyTitle><EmptyDescription>This chapter's video is still being rendered. The ideas and lab below are ready.</EmptyDescription></EmptyHeader></Empty>
      ))}

      {ch.lab && (
        <a href={ch.lab.href} className="group flex items-center gap-3 rounded-xl border bg-gradient-to-r from-viz-a/10 to-card p-4 transition-colors hover:border-foreground/20">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground"><FlaskConical className="size-5" /></div>
          <div className="min-w-0 flex-1"><div className="text-xs text-muted-foreground">Try it yourself</div><div className="font-medium">{ch.lab.label}</div></div>
          <ArrowRight className="size-5 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
        </a>
      )}

      {!content ? <p className="text-sm text-muted-foreground">Key ideas and checks for this chapter are being written.</p> : (<>
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-lg font-semibold tracking-tight"><Lightbulb className="size-5 text-pin" />Key ideas</h2>
          <div className="grid gap-3 @2xl:grid-cols-2">
            {content.keyIdeas.map((k, i) => (
              <Card key={i} className="gap-2 py-4" data-ask-id={`idea-${i}`} data-ask={`Key idea "${k.title}": ${k.body}`}>
                <CardHeader className="px-4"><CardTitle className="flex items-start gap-2.5 text-base leading-snug"><span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-xs tabular-nums">{i + 1}</span>{k.title}</CardTitle></CardHeader>
                <CardContent className="space-y-2 px-4 text-sm text-muted-foreground">
                  <p className="text-pretty">{k.body}</p>
                  {k.tryIt && <p className="rounded-md bg-muted/60 px-2.5 py-1.5 text-xs"><span className="font-medium text-foreground">Try it: </span>{k.tryIt}</p>}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {content.pitfall && (
          <Alert data-ask-id="pitfall" data-ask={`Common misconception: ${content.pitfall.myth}. Truth: ${content.pitfall.truth}`}>
            <TriangleAlert />
            <AlertTitle>Common misconception</AlertTitle>
            <AlertDescription><p><span className="line-through decoration-destructive/60">{content.pitfall.myth}</span></p><p className="text-foreground">{content.pitfall.truth}</p></AlertDescription>
          </Alert>
        )}

        <Quiz chapter={ch.id} questions={content.quiz} />

        <Collapsible defaultOpen={level === "advanced"} className="group/deep rounded-xl border bg-card">
          <CollapsibleTrigger className="flex w-full items-center gap-3 p-4 text-left">
            <div className="flex size-8 items-center justify-center rounded-lg bg-muted"><GraduationCap className="size-4" /></div>
            <div className="flex-1"><div className="font-medium">Go deeper: the math and the code</div><div className="text-xs text-muted-foreground">For when you want the exact mechanism</div></div>
            <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[state=open]/deep:rotate-180" />
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-4 border-t p-4">
            <pre className="overflow-x-auto rounded-lg bg-muted/50 p-3 font-mono text-[13px] leading-relaxed whitespace-pre-wrap" data-ask-id="deeper-math" data-ask={`The math for ${ch.title}: ${content.deeper.math}`}>{content.deeper.math}</pre>
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm text-muted-foreground">{content.deeper.codeCaption}</p>
                <Button size="xs" variant="outline" onClick={() => { navigator.clipboard.writeText(content.deeper.code); toast.success("Code copied"); }}><Copy />Copy</Button>
              </div>
              <pre className="max-h-[420px] overflow-auto rounded-lg border bg-[var(--viz-stage)] p-3 font-mono text-[12.5px] leading-relaxed" data-ask-id="deeper-code" data-ask={`Code example for ${ch.title}:\n${content.deeper.code}`}><code>{content.deeper.code}</code></pre>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {content.further?.length ? (
          <section className="space-y-2">
            <h2 className="text-sm font-medium text-muted-foreground">Further reading & watching</h2>
            <div className="grid gap-2 @2xl:grid-cols-2">
              {content.further.map((f) => (
                <a key={f.url} href={f.url} target="_blank" rel="noreferrer" className="flex items-start gap-3 rounded-lg border p-3 text-sm transition-colors hover:bg-muted/50">
                  <ExternalLink className="mt-0.5 size-4 shrink-0 text-muted-foreground" /><span><span className="font-medium">{f.title}</span><span className="block text-xs text-muted-foreground">{f.note}</span></span>
                </a>
              ))}
            </div>
          </section>
        ) : null}
      </>)}

      <div className="flex items-stretch gap-2 border-t pt-4">
        {prev ? <Button variant="outline" className="h-auto flex-1 justify-start py-2 text-left whitespace-normal" asChild><a href={`#/learn/${prev.id}`}><ArrowLeft /><span><span className="block text-xs text-muted-foreground">Previous</span>{prev.title}</span></a></Button> : <span className="flex-1" />}
        {next ? <Button className="h-auto flex-1 justify-end py-2 text-right whitespace-normal" asChild onClick={() => setProgress(ch.id, { done: true })}><a href={`#/learn/${next.id}`}><span><span className="block text-xs opacity-70">Next</span>{next.title}</span><ArrowRight /></a></Button>
          : <Button className="flex-1" asChild onClick={() => setProgress(ch.id, { done: true })}><a href="#/invent"><Check />Finish: go invent something</a></Button>}
      </div>
    </div>
  );
}

function Quiz({ chapter, questions }: { chapter: string; questions: { q: string; options: string[]; answer: number; why: string }[] }) {
  const [picked, setPicked] = useState<(number | null)[]>(() => questions.map(() => null));
  const score = useMemo(() => picked.filter((p, i) => p === questions[i].answer).length, [picked, questions]);
  const answered = picked.filter((p) => p !== null).length;
  useEffect(() => { if (answered === questions.length) setProgress(chapter, { quiz: score }); }, [answered, score, chapter, questions.length]);
  return (
    <section className="space-y-3">
      <div className="flex items-end justify-between gap-2">
        <h2 className="flex items-center gap-2 text-lg font-semibold tracking-tight"><CheckCircle2 className="size-5 text-viz-b" />Check yourself</h2>
        {answered === questions.length && <Badge variant={score >= 2 ? "default" : "secondary"}>{score}/{questions.length} right</Badge>}
      </div>
      <div className="space-y-3">
        {questions.map((q, qi) => {
          const p = picked[qi];
          return (
            <Card key={qi} className="gap-3 py-4" data-ask-id={`quiz-${qi}`} data-ask={`Quiz question: ${q.q} Options: ${q.options.join(" / ")}${p !== null ? `. The learner picked "${q.options[p]}" (${p === q.answer ? "correct" : `wrong, correct is "${q.options[q.answer]}"`})` : ""}`}>
              <CardHeader className="px-4"><CardTitle className="text-sm leading-snug font-medium">{qi + 1}. {q.q}</CardTitle><CardDescription className="sr-only">Pick one answer</CardDescription></CardHeader>
              <CardContent className="space-y-2 px-4">
                <div className="grid gap-1.5">
                  {q.options.map((o, oi) => {
                    const state = p === null ? "idle" : oi === q.answer ? "right" : oi === p ? "wrong" : "dim";
                    return (
                      <button key={oi} disabled={p !== null} onClick={() => setPicked(picked.map((x, i) => (i === qi ? oi : x)))}
                        className={cn("flex min-h-10 items-center gap-2.5 rounded-lg border px-3 py-2 text-left text-sm transition-colors",
                          state === "idle" && "hover:bg-muted/60", state === "right" && "border-emerald-500/60 bg-emerald-500/10",
                          state === "wrong" && "border-destructive/60 bg-destructive/10", state === "dim" && "opacity-60")}>
                        <span className="flex size-5 shrink-0 items-center justify-center rounded-full border text-[11px]">{state === "right" ? <Check className="size-3" /> : String.fromCharCode(65 + oi)}</span>{o}
                      </button>
                    );
                  })}
                </div>
                {p !== null && <p className={cn("rounded-md px-3 py-2 text-sm", p === q.answer ? "bg-emerald-500/10" : "bg-muted")}>{p === q.answer ? "Right. " : "Not quite. "}{q.why}</p>}
              </CardContent>
            </Card>
          );
        })}
      </div>
      {answered > 0 && <Button variant="ghost" size="sm" onClick={() => setPicked(questions.map(() => null))}>Try again</Button>}
    </section>
  );
}
