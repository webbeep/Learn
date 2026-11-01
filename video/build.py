#!/usr/bin/env python3
"""Render the Top-Down AI explainer series and publish it to public/videos/.

    python video/build.py                 # render all 5 at 1080p30, encode, posters, index.json
    python video/build.py tokens attention  # only some ids (index.json keeps the others' entries)
    python video/build.py --preview       # fast 480p15 render (into video/media only, nothing published)
    python video/build.py --index-only    # rebuild public/videos/index.json from video/build/*.json

Needs: manim 0.22 (no LaTeX), ffmpeg/ffprobe, fonts Inter + JetBrains Mono, torch, and the env vars
CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN for narration (cached in video/.cache by hash of voice+text).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "public", "videos")
MAX_BYTES = 24 * 1024 * 1024 - 256 * 1024  # stay under Cloudflare Pages' 25 MiB/file with margin

VIDEOS = [
    dict(id="gpt-in-100s", file="v1_gpt.py", cls="GPTIn100", relates="xray", title="GPT in 100 seconds",
         subtitle="The whole pipeline on a real tiny GPT: tokens → vectors → attention → probabilities → loop",
         poster=("Attention", 1, 0.75)),
    dict(id="tokens", file="v2_tokens.py", cls="Tokens", relates="xray", title="Tokens: how a model reads",
         subtitle="Byte-pair encoding, merge by merge, with TinyGPT's real 256-token vocab",
         poster=("Encoding a word", 1, 0.97)),
    dict(id="attention", file="v3_attention.py", cls="Attention", relates="xray", title="Attention, visually",
         subtitle="Queries, keys, values, softmax — and a real head that copies colors",
         poster=("A real head", 0, 0.95)),
    dict(id="next-token", file="v4_next.py", cls="NextToken", relates="xray",
         title="Next token: logits, softmax & temperature",
         subtitle="How 256 raw scores become one sampled word", poster=("Temperature", 2, 0.97)),
    dict(id="training", file="v5_training.py", cls="Training", relates="training",
         title="How it learned (training in 100 seconds)",
         subtitle="Loss, gradient descent, and real samples from step 0 to step 3000",
         poster=("The loss curve", 0, 0.97)),
]


def run(cmd, **kw):
    print("$", " ".join(cmd[:6]), "…" if len(cmd) > 6 else "", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def probe_dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                          path], capture_output=True, text=True, check=True).stdout
    return float(out)


def render(v, preview=False):
    q = ["-ql"] if preview else ["-r", "1920,1080", "--fps", "30"]
    log = os.path.join(HERE, "build", f"{v['id']}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        run(["manim", *q, "--disable_caching", "--media_dir", os.path.join(HERE, "media"),
             os.path.join(HERE, "scenes", v["file"]), v["cls"]], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    res = "480p15" if preview else "1080p30"
    mod = os.path.splitext(v["file"])[0]
    path = os.path.join(HERE, "media", "videos", mod, res, f"{v['cls']}.mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    for line in open(log):
        if "overran" in line:
            print(f"[{v['id']}] {line.strip()}")
    return path


def encode(v, src):
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"{v['id']}.mp4")
    for crf in (24, 26, 28, 30, 33):
        run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-tune", "animation", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst])
        size = os.path.getsize(dst)
        print(f"[{v['id']}] crf {crf}: {size / 1e6:.2f} MB")
        if size < MAX_BYTES:
            return dst
    raise RuntimeError(f"{v['id']} too large even at crf 33")


def poster(v, mp4, meta):
    chap, seg_off, frac = v["poster"]
    t0 = next(c["t"] for c in meta["chapters"] if c["title"] == chap)
    i = next(k for k, s in enumerate(meta["segments"]) if abs(s["start"] - t0) < 1e-6) + seg_off
    s = meta["segments"][i]
    t = s["start"] + (s["end"] - s["start"]) * frac
    dst = os.path.join(OUT, f"{v['id']}.jpg")
    for q in (3, 5, 7, 9, 12):
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", mp4, "-frames:v", "1", "-vf", "scale=1280:720",
             "-q:v", str(q), dst])
        if os.path.getsize(dst) < 200 * 1024:
            break
    return dst


def write_index():
    entries = []
    for v in VIDEOS:
        p = os.path.join(HERE, "build", f"{v['id']}.json")
        if not os.path.exists(p) or not os.path.exists(os.path.join(OUT, f"{v['id']}.mp4")):
            print(f"[index] skipping {v['id']} (not built)")
            continue
        m = json.load(open(p))
        dur = round(probe_dur(os.path.join(OUT, f"{v['id']}.mp4")), 2)
        entries.append({
            "id": v["id"], "title": v["title"], "subtitle": v["subtitle"], "duration": dur,
            "src": f"/videos/{v['id']}.mp4", "poster": f"/videos/{v['id']}.jpg", "relates": v["relates"],
            "voice": m["voice"], "chapters": m["chapters"],
            "segments": [{k: s[k] for k in ("start", "end", "text", "scene", "objects")} for s in m["segments"]],
        })
        last = m["segments"][-1]["end"]
        print(f"[index] {v['id']}: mp4 {dur:.2f}s, last segment ends {last:.2f}s, {len(m['segments'])} segments")
    json.dump(entries, open(os.path.join(OUT, "index.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote", os.path.join(OUT, "index.json"))


def build_one(v, preview):
    src = render(v, preview)
    if preview:
        return
    meta = json.load(open(os.path.join(HERE, "build", f"{v['id']}.json")))
    mp4 = encode(v, src)
    poster(v, mp4, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if not a.index_only:
        todo = [v for v in VIDEOS if not a.ids or v["id"] in a.ids]
        with ThreadPoolExecutor(a.jobs) as ex:
            for f in [ex.submit(build_one, v, a.preview) for v in todo]:
                f.result()
    if not a.preview:
        write_index()


if __name__ == "__main__":
    sys.exit(main())
