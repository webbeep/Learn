import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { ThemeProvider, useTheme } from "next-themes";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Spinner } from "@/components/ui/spinner";
import { useIsMobile } from "@/hooks/use-mobile";
import { AppSidebar, NAV, isActive } from "@/components/app/app-sidebar";
import { SiteHeader } from "@/components/app/site-header";
import { Onboarding } from "@/components/app/onboarding";
import { MobileNav } from "@/components/app/mobile-nav";
import { loadModel, type TinyGPT } from "./model/gpt";
import { ask, markVisited, useStore } from "./lib/store";
import { setVizDark } from "./lib/theme";
import { AskPopover, TutorPanel } from "./ask/AskPanel";
import { MarkStyles, RegionLayer, useClickToAsk } from "./ask/Interact";
import { ShortcutHelp, VoiceHUD, useShortcuts } from "./ask/Shortcuts";
import { trackPointer, trackTextSelection } from "./lib/conversation";
const XRay = lazy(() => import("./pages/XRay").then((m) => ({ default: m.XRay })));
const Training = lazy(() => import("./pages/Training").then((m) => ({ default: m.Training })));
const Videos = lazy(() => import("./pages/Videos").then((m) => ({ default: m.Videos })));
import { MapPage } from "./pages/MapPage";
const Path = lazy(() => import("./pages/Path").then((m) => ({ default: m.Path })));
const Learn = lazy(() => import("./pages/Learn").then((m) => ({ default: m.Learn })));
const ChapterPage = lazy(() => import("./pages/Learn").then((m) => ({ default: m.ChapterPage })));
const TrainLab = lazy(() => import("./pages/TrainLab").then((m) => ({ default: m.TrainLab })));
const ArchLab = lazy(() => import("./pages/ArchLab").then((m) => ({ default: m.ArchLab })));
const Backprop = lazy(() => import("./pages/Backprop").then((m) => ({ default: m.Backprop })));
const Scaling = lazy(() => import("./pages/Scaling").then((m) => ({ default: m.Scaling })));
const Invent = lazy(() => import("./pages/Invent").then((m) => ({ default: m.Invent })));

function useHash() {
  const [h, setH] = useState(location.hash || "#/");
  useEffect(() => { const f = () => setH(location.hash || "#/"); window.addEventListener("hashchange", f); return () => window.removeEventListener("hashchange", f); }, []);
  return h === "#/watch" ? "#/videos" : h;
}

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
      <TooltipProvider delayDuration={300}>
        <Shell />
        <Toaster position="top-center" />
      </TooltipProvider>
    </ThemeProvider>
  );
}

function Shell() {
  const hash = useHash();
  const { resolvedTheme } = useTheme();
  setVizDark(resolvedTheme !== "light"); // JS-computed viz colors follow the theme
  const [model, setModel] = useState<TinyGPT | null>(null);
  const [err, setErr] = useState("");
  const [help, setHelp] = useState(false);
  const main = useRef<HTMLElement>(null);
  const panelOpen = useStore(ask, (s) => s.open);
  const isMobile = useIsMobile();
  useClickToAsk(main);
  useShortcuts(setHelp);
  useEffect(() => { const a = trackPointer(), b = trackTextSelection(); return () => { a(); b(); }; }, []);
  useEffect(() => { loadModel().then(setModel).catch((e) => setErr(String(e))); }, []);
  const pageKey = hash.split("/").slice(0, 3).join("/");
  useEffect(() => { markVisited(hash); ask.set({ selection: [], region: "", highlight: null, scene: "" }); window.scrollTo(0, 0); }, [pageKey]); // eslint-disable-line

  const route = (h: string, base: string) => h === base || h.startsWith(base + "/") || h.startsWith(base + "?");
  const needsModel = route(hash, "#/xray") || route(hash, "#/training");
  let page: React.ReactNode;
  if (needsModel && !model) page = err
    ? <div className="p-10 text-center text-destructive">Couldn't load the model: {err}</div>
    : <div className="flex items-center justify-center gap-2 p-16 text-muted-foreground"><Spinner />Loading TinyGPT (680 KB of weights)…</div>;
  else if (hash.startsWith("#/xray")) page = <XRay key={hash} m={model!} initialScene={hash.split("/")[2]} />;
  else if (route(hash, "#/training")) page = <Training corpus={model!.meta.corpus_sample} />;
  else if (hash.startsWith("#/videos")) page = <Videos />;
  else if (hash.startsWith("#/path")) page = <Path />;
  else if (hash.startsWith("#/learn/")) page = <ChapterPage key={hash} id={hash.split("/")[2]} />;
  else if (hash.startsWith("#/learn")) page = <Learn />;
  else if (route(hash, "#/train")) page = <TrainLab />;
  else if (route(hash, "#/lab")) page = <ArchLab />;
  else if (route(hash, "#/backprop")) page = <Backprop />;
  else if (route(hash, "#/scaling")) page = <Scaling />;
  else if (route(hash, "#/invent")) page = <Invent />;
  else page = <MapPage />;
  const title = NAV.find((n) => isActive(n.href, hash))?.label ?? "Home";

  return (
    <SidebarProvider>
      <AppSidebar hash={hash} onShortcuts={() => setHelp(true)} />
      <SidebarInset className="min-w-0">
        <SiteHeader title={title} />
        <div className="flex min-h-0 flex-1">
          <div ref={main as React.RefObject<HTMLDivElement>} key={resolvedTheme} className="min-w-0 flex-1 px-3 pt-4 pb-24 sm:px-4 md:px-6 md:pt-5 md:pb-5 lg:px-8">
            <div className="mx-auto w-full max-w-6xl"><Suspense fallback={<div className="flex items-center justify-center gap-2 p-16 text-muted-foreground"><Spinner />Loading…</div>}>{page}</Suspense></div>
          </div>
          {panelOpen && !isMobile && (
            <aside className="sticky top-14 h-[calc(100svh-3.5rem)] w-[380px] shrink-0 border-l xl:w-[420px]"><TutorPanel /></aside>
          )}
        </div>
      </SidebarInset>
      {isMobile && (
        <Sheet open={panelOpen} onOpenChange={(o) => ask.set({ open: o })}>
          <SheetContent side="right" className="w-full gap-0 p-0 data-[side=right]:w-full data-[side=right]:sm:max-w-md pb-[env(safe-area-inset-bottom)] sm:max-w-md [&>button.absolute]:hidden">
            <SheetTitle className="sr-only">Tutor</SheetTitle>
            <TutorPanel />
          </SheetContent>
        </Sheet>
      )}
      <AskPopover />
      <RegionLayer main={main} />
      <MarkStyles />
      <VoiceHUD />
      <ShortcutHelp open={help} onOpenChange={setHelp} />
      <Onboarding />
      <MobileNav hash={hash} />
    </SidebarProvider>
  );
}
