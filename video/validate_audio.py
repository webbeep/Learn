"""Validate the narration of every published video, end to end, from the final MP4s.

For each narration segment (times from public/videos/index.json):
  * naturalness: UTMOS predicted MOS (1-5) on the segment audio
  * intelligibility: Whisper (Workers AI) transcript vs the caption text -> word error rate
  * pace: words per minute over the segment
For each video: longest silence (dead air), integrated loudness (LUFS), and any narration that
animations overran during the render (from the build log, if given with --log).

Usage: python3 video/validate_audio.py --utmos /path/utmos_cpu.jit [--log render.log] [ids...]
Writes video/build/audio_report.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("VALIDATE_ROOT", os.path.dirname(HERE))


def pcm16(path, start=None, end=None, sr=16000):
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", f"{start:.2f}", "-to", f"{end:.2f}"] if start is not None else []) + \
          ["-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.int16)


def wav_bytes(x, sr=16000):
    import io, wave
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())
    return b.getvalue()


def transcribe(x):
    url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/run/@cf/openai/whisper-large-v3-turbo"
    body = json.dumps({"audio": base64.b64encode(wav_bytes(x)).decode(), "language": "en"}).encode()
    for _ in range(4):
        try:
            req = urllib.request.Request(url, data=body, method="POST", headers={
                "Authorization": "Bearer " + os.environ["CLOUDFLARE_API_TOKEN"], "content-type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=120))["result"]["text"]
        except Exception:  # noqa: BLE001
            pass
    return ""


NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def norm(s):
    s = s.lower().replace("%", " percent").replace("tinygpt", "tiny gpt").replace("—", " ").replace("-", " ")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    toks = re.sub(r"[^a-z0-9. ]", " ", s).split()
    return [NUMW.get(t.strip("."), t.strip(".")) for t in toks if t.strip(".")]


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int); d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(1, len(r))


def longest_silence(path):
    out = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", "silencedetect=noise=-45dB:d=0.8", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    durs = [float(x) for x in re.findall(r"silence_duration: ([\d.]+)", out)]
    return max(durs) if durs else 0.0, len(durs)


def loudness(path):
    out = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", out)
    return float(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--utmos", required=True)
    ap.add_argument("--log")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(3)
    model = torch.jit.load(a.utmos, map_location="cpu").eval()
    overruns = {}
    if a.log and os.path.exists(a.log):
        for line in open(a.log):
            m = re.search(r"overran audio by ([\d.]+)s: '(.{0,40})", line)
            if m:
                overruns.setdefault(m.group(2)[:30], float(m.group(1)))
    index = json.load(open(os.path.join(ROOT, "public", "videos", "index.json")))
    report = []
    for v in index:
        if a.ids and v["id"] not in a.ids:
            continue
        mp4 = os.path.join(ROOT, "public", v["src"].lstrip("/"))
        segs = v["segments"]
        clips = [pcm16(mp4, s["start"], s["end"]) for s in segs]
        with ThreadPoolExecutor(6) as ex:
            hyps = list(ex.map(transcribe, clips))
        rows = []
        for s, x, hyp in zip(segs, clips, hyps):
            scores = []
            for i in range(0, max(1, len(x) - 32000), 160000):
                seg = x[i:i + 160000]
                if len(seg) >= 32000:
                    with torch.no_grad():
                        scores.append(model(torch.tensor(seg).unsqueeze(0)).item())
            words = len(s["text"].split())
            rows.append({"start": s["start"], "utmos": round(float(np.mean(scores)), 2) if scores else None,
                         "wer": round(wer(s["text"], hyp), 3), "wpm": round(words / max(0.1, s["end"] - s["start"]) * 60),
                         "text": s["text"][:90], "heard": hyp[:90]})
        sil, nsil = longest_silence(mp4)
        rep = {"id": v["id"], "duration": v["duration"], "utmos_mean": round(float(np.mean([r["utmos"] for r in rows if r["utmos"]])), 2),
               "utmos_min": min(r["utmos"] for r in rows if r["utmos"]), "wer_mean": round(float(np.mean([r["wer"] for r in rows])), 3),
               "wer_max": max(r["wer"] for r in rows), "wpm_mean": round(float(np.mean([r["wpm"] for r in rows]))),
               "longest_silence": round(sil, 2), "lufs": loudness(mp4), "segments": rows}
        report.append(rep)
        print(f"{v['id']:22s} utmos {rep['utmos_mean']:.2f} (min {rep['utmos_min']:.2f})  wer {rep['wer_mean']:.3f} (max {rep['wer_max']:.2f})  "
              f"{rep['wpm_mean']} wpm  longest silence {rep['longest_silence']}s  {rep['lufs']} LUFS", flush=True)
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    json.dump({"videos": report, "overruns": overruns}, open(os.path.join(HERE, "build", "audio_report.json"), "w"), indent=1)
    worst = sorted((r for v in report for r in v["segments"]), key=lambda r: -r["wer"])[:8]
    print("\nhighest-WER segments (check pronunciation):")
    for r in worst:
        print(f"  {r['wer']:.2f}  said: {r['text']!r}\n        heard: {r['heard']!r}")
    print(f"\nanimation overruns during render: {len(overruns)}")


if __name__ == "__main__":
    sys.exit(main())
