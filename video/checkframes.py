"""Extract a frame at the middle (or given fraction) of each segment and draw the clickable-object boxes on it.
usage: python checkframes.py <video.mp4> <build json> [frac]"""
import json, subprocess, sys, os
from PIL import Image, ImageDraw
mp4, js = sys.argv[1], sys.argv[2]
frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
d = json.load(open(js))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames", d["id"])
os.makedirs(out, exist_ok=True)
for n, s in enumerate(d["segments"]):
    t = s["start"] + (s["end"] - s["start"]) * frac
    png = os.path.join(out, f"{n:02d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", mp4, "-frames:v", "1", png], check=True)
    im = Image.open(png).convert("RGB"); W, H = im.size; dr = ImageDraw.Draw(im)
    for o in s["objects"]:
        x, y, w, h = o["box"]
        dr.rectangle([x * W, y * H, (x + w) * W, (y + h) * H], outline=(0, 255, 0), width=1)
    im.save(os.path.join(out, f"{n:02d}_boxes.png"))
    print(png, f"t={t:.1f}", len(s["objects"]))
