"""Shared base for the Top-Down AI explainer videos.

* VScene: dark 3b1b-style Scene with the series palette and fonts.
* self.say(text, scene=..., chapter=...): narrated block (Cloudflare Workers AI
  Deepgram Aura-2 TTS, cached in video/.cache). Animations played inside the
  block run while the line is spoken; the block then waits out the audio.
* self.mark(id, mobject, label): register a clickable on-screen object. At the
  end of each say-block every registered mobject that is on screen is
  snapshotted (normalised [x, y, w, h], top-left origin) into the segment.
* On tear-down the timeline is written to video/build/<VIDEO_ID>.json, which
  build.py merges into public/videos/index.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.request
from contextlib import contextmanager

from manim import *  # noqa: F401,F403
from manim import config

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
BUILD = os.path.join(HERE, "build")

VOICE = os.environ.get("VIDEO_VOICE", "orion")
TEMPO = 1.08          # slight speed-up for Fireship pacing
AUDIO_PIPE = "v2"   # bump when the post-processing chain below changes
GAP = 0.22            # breathing room after each line

# ---- palette ---------------------------------------------------------------
BG = "#0b0e17"
BLUE = "#7c9cff"
TEAL = "#48d6b0"
YELLOW = "#ffd54a"
ORANGE = "#ff9640"
PINK = "#ff6b9a"
MUTED = "#8b93ab"
FG = "#e6e9f2"
PANEL = "#151a28"
LINE = "#2a3147"
CYCLE = [BLUE, TEAL, ORANGE, PINK, YELLOW, "#b48cff", "#5fd0ff", "#c3e66e"]
SANS = "Inter"
MONO = "JetBrains Mono"


# ---- TTS ---------------------------------------------------------------------
def _ffprobe_dur(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=nw=1:nk=1", path], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def tts(text: str, voice: str = VOICE) -> tuple[str, float]:
    """Return (wav_path, duration) for a narration line, generating + caching on demand."""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{TEMPO}|{text}".encode()).hexdigest()[:20]
    wav = os.path.join(CACHE, f"{key}_{AUDIO_PIPE}.wav")
    if not os.path.exists(wav):
        raw = os.path.join(CACHE, f"{key}.mp3")
        if not os.path.exists(raw):
            acct = os.environ["CLOUDFLARE_ACCOUNT_ID"]
            url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/deepgram/aura-2-en"
            body = json.dumps({"text": text, "speaker": voice}).encode()
            for attempt in range(5):
                try:
                    req = urllib.request.Request(url, data=body, method="POST", headers={
                        "Authorization": "Bearer " + os.environ["CLOUDFLARE_API_TOKEN"],
                        "content-type": "application/json"})
                    with urllib.request.urlopen(req, timeout=90) as r:
                        data = r.read()
                    if len(data) < 1000:
                        raise RuntimeError("tiny TTS response")
                    open(raw, "wb").write(data)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[tts] retry {attempt + 1}: {type(e).__name__}")
                    time.sleep(2 + 3 * attempt)
            else:
                raise RuntimeError("TTS failed for: " + text[:60])
        trim = "silenceremove=start_periods=1:start_threshold=-50dB"
        # cap long TTS pauses (colons, list items) at ~0.6 s, trim both ends, speed up slightly
        cap = "silenceremove=stop_periods=-1:stop_duration=0.3:stop_threshold=-40dB:stop_silence=0.3:detection=peak"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af",
                        f"{trim},{cap},areverse,{trim},areverse,atempo={TEMPO}", "-ar", "48000", "-ac", "1", wav],
                       check=True)
    return wav, _ffprobe_dur(wav)


class Narration:
    """Handle yielded by VScene.say()."""

    def __init__(self, scene, text, start, dur):
        self.scene, self.text, self.start, self.dur = scene, text, start, dur

    @property
    def elapsed(self):
        return self.scene.renderer.time - self.start

    @property
    def left(self):
        return max(0.0, self.dur - self.elapsed)

    def t_of(self, phrase: str) -> float:
        i = self.text.lower().find(phrase.lower())
        if i < 0:
            raise ValueError(f"phrase {phrase!r} not in line")
        return self.dur * i / max(1, len(self.text))


# ---- base scene --------------------------------------------------------------
class VScene(Scene):
    VIDEO_ID = "video"

    def setup(self):
        self.camera.background_color = BG
        self._segments, self._chapters, self._marks = [], [], {}
        self._line = None

    # -- narration ---------------------------------------------------------------
    @contextmanager
    def say(self, text: str, scene: str = "", chapter: str | None = None, gap: float = GAP):
        path, dur = tts(text)
        start = self.renderer.time
        self.add_sound(path)
        if chapter:
            self._chapters.append({"t": round(start, 2), "title": chapter})
        line = Narration(self, text, start, dur)
        self._line = line
        yield line
        rest = dur + gap - line.elapsed
        if rest > 0.02:
            self.wait(rest)
        elif rest < -0.3:
            print(f"[say] animations overran audio by {-rest:.2f}s: {text[:50]!r}")
        end = self.renderer.time
        self._segments.append({"start": round(start, 2), "end": round(end, 2), "text": text,
                               "scene": scene, "objects": self._snapshot()})
        self._line = None

    def sync(self, phrase: str, lead: float = 0.15):
        """Wait until (approximately) `phrase` is being spoken in the current line."""
        ln = self._line
        target = ln.t_of(phrase) - lead
        if target - ln.elapsed > 0.02:
            self.wait(target - ln.elapsed)

    def pause(self, t: float):
        self.wait(t)

    # -- clickable objects -------------------------------------------------------
    def mark(self, mid: str, mob: Mobject, label: str):
        self._marks[mid] = (mob, label)
        return mob

    def unmark(self, *ids):
        for i in ids:
            self._marks.pop(i, None)

    def _snapshot(self):
        fam = set(self.get_mobject_family_members())
        fw, fh = config.frame_width, config.frame_height
        objs = []
        def on_screen(m):
            return m in fam or (len(m.submobjects) > 0 and all(on_screen(x) for x in m.submobjects))

        for mid, (mob, label) in self._marks.items():
            if not on_screen(mob):
                continue
            leaves = [m for m in mob.get_family() if m.has_points()]
            vis = False
            for m in leaves:
                if isinstance(m, VMobject):
                    if m.get_fill_opacity() > 0.02 or (m.get_stroke_opacity() > 0.02 and m.get_stroke_width() > 0):
                        vis = True; break
                else:
                    vis = True; break
            if not vis:
                continue
            ul, dr = mob.get_corner(UL), mob.get_corner(DR)
            x0 = max(0.0, (ul[0] + fw / 2) / fw); x1 = min(1.0, (dr[0] + fw / 2) / fw)
            y0 = max(0.0, (fh / 2 - ul[1]) / fh); y1 = min(1.0, (fh / 2 - dr[1]) / fh)
            if x1 - x0 < 0.003 or y1 - y0 < 0.003:
                continue
            objs.append({"id": mid, "label": label,
                         "box": [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]})
        if len(objs) > 40:
            print(f"[mark] {len(objs)} objects on screen, truncating to 40")
            objs = objs[:40]
        return objs

    def tear_down(self):
        os.makedirs(BUILD, exist_ok=True)
        json.dump({"id": self.VIDEO_ID, "voice": VOICE, "duration": round(self.renderer.time, 2),
                   "chapters": self._chapters, "segments": self._segments},
                  open(os.path.join(BUILD, f"{self.VIDEO_ID}.json"), "w"), indent=1)
        super().tear_down()

    # -- building blocks -----------------------------------------------------------
    @staticmethod
    def T(s, size=30, color=FG, weight=NORMAL, font=SANS, **kw):
        return Text(s, font=font, font_size=size, color=color, weight=weight, **kw)

    @staticmethod
    def M(s, size=28, color=FG, **kw):
        return Text(s, font=MONO, font_size=size, color=color, **kw)

    @staticmethod
    def show_tok(s: str) -> str:
        return s.replace(" ", "␣").replace("\n", "↵")

    def chip(self, tok: str, color=BLUE, size=28, pad=0.12, id_=None, id_size=20, fill=0.16, font=MONO):
        """A token chip: rounded box + mono label (leading space shown as ␣); optional id underneath."""
        disp = self.show_tok(tok)
        if font != MONO:
            disp = tok
        lab = Text(disp, font=font, font_size=size, color=FG, t2c={"␣": MUTED, "↵": MUTED})
        h = max(Text("Hg", font=font, font_size=size).height + 2 * pad, 0.1)
        box = RoundedRectangle(corner_radius=0.08, width=lab.width + 2 * pad + 0.04, height=h,
                               stroke_color=color, stroke_width=2.2, fill_color=color, fill_opacity=fill)
        lab.move_to(box)
        g = VGroup(box, lab)
        if id_ is not None:
            idt = Text(str(id_), font=MONO, font_size=id_size, color=color)
            idt.next_to(box, DOWN, buff=0.1)
            g.add(idt)
        return g

    def live(self, fn, place, size=40, color=YELLOW, font=MONO, weight=NORMAL):
        """LaTeX-free live number: fn() -> str, place(mobject) positions it. Redrawn every frame."""
        def make():
            t = Text(fn(), font=font, font_size=size, color=color, weight=weight)
            place(t)
            return t
        return always_redraw(make)

    def flash(self, text, color=YELLOW, size=64, hold=0.7, pos=ORIGIN, bg=True):
        """Fireship-style punchline: pop in, hold, pop out."""
        t = Text(text, font=SANS, font_size=size, color=color, weight=BOLD).move_to(pos)
        grp = VGroup(t)
        if bg:
            plate = RoundedRectangle(corner_radius=0.15, width=t.width + 0.8, height=t.height + 0.5,
                                     fill_color=BG, fill_opacity=0.92, stroke_color=color, stroke_width=3)
            plate.move_to(t)
            grp = VGroup(plate, t)
        self.play(FadeIn(grp, scale=1.35), run_time=0.25)
        self.wait(hold)
        self.play(FadeOut(grp, scale=0.85), run_time=0.2)

    def header(self, text, color=MUTED):
        t = Text(text.upper(), font=SANS, font_size=24, color=color, weight=BOLD)
        t.to_corner(UL, buff=0.45)
        return t

    def clear_all(self, run_time=0.5, keep=()):
        mobs = [m for m in self.mobjects if m not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)
