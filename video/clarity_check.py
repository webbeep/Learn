"""Check hand-written clarity rewrites in video/clarity/<id>.json against the hard rules.

Each file: {"id": "<video id>", "lines": [{"orig": "...", "new": "..."}, ...]} in segment order.
Rules per line: same numbers in the same order; every sync phrase that occurs in the original line
appears word for word in the new line, in the same order; length 0.9x-1.85x of the original.
Also prints a readability estimate (Flesch-Kincaid grade) before -> after.

Usage: python3 video/clarity_check.py [ids...]          (exit code 1 if any line fails)
       python3 video/clarity_check.py --template <id>   (print a skeleton file for a video)
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import narrate  # noqa: E402
from clarify import check, keeps_for  # noqa: E402

ROOT = os.path.dirname(HERE)


def originals():
    cache = json.load(open(narrate.NATURAL_PATH)) if os.path.exists(narrate.NATURAL_PATH) else {}
    back = {v: k for k, v in cache.items()}
    idx = json.load(open(os.path.join(ROOT, "public", "videos", "index.json")))
    return {v["id"]: [{"orig": back.get(s["text"], s["text"]), "scene": s.get("scene", "")} for s in v["segments"]] for v in idx}


def syllables(w):
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w: return 0
    n = len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and not w.endswith("le") else 0)
    return max(1, n)


def fk(text):
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z0-9']+", text)
    if not words: return 0
    return 0.39 * len(words) / sents + 11.8 * sum(syllables(w) for w in words) / len(words) - 15.59


def main():
    orig = originals()
    ph = lambda vid: narrate._sync_phrases(narrate.scene_file(vid))
    if sys.argv[1:2] == ["--template"]:
        vid = sys.argv[2]; phrases = ph(vid)
        print(json.dumps({"id": vid, "lines": [{"orig": o["orig"], "visuals": o["scene"], "keep": keeps_for(o["orig"], phrases), "new": ""} for o in orig[vid]]}, indent=1, ensure_ascii=False))
        return 0
    if sys.argv[1:2] == ["--apply"]:
        # copy every passing rewrite into narration_natural.json (original line -> spoken line)
        cache = json.load(open(narrate.NATURAL_PATH)) if os.path.exists(narrate.NATURAL_PATH) else {}
        n = 0
        for f in sorted(os.listdir(os.path.join(HERE, "clarity"))):
            if not f.endswith(".json"): continue
            d = json.load(open(os.path.join(HERE, "clarity", f))); phrases = ph(d["id"])
            for l in d["lines"]:
                new, why = check(l["orig"], l.get("new", ""), keeps_for(l["orig"], phrases))
                if new is not None:
                    cache[l["orig"]] = new; n += 1
        json.dump(cache, open(narrate.NATURAL_PATH, "w"), indent=1, ensure_ascii=False)
        print("applied", n, "lines"); return 0
    ids = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(os.path.join(HERE, "clarity")) if f.endswith(".json"))
    bad = 0
    for vid in ids:
        path = os.path.join(HERE, "clarity", f"{vid}.json")
        if not os.path.exists(path):
            print(f"{vid}: missing"); bad += 1; continue
        d = json.load(open(path))
        want = [o["orig"] for o in orig[vid]]
        got = [l["orig"] for l in d["lines"]]
        if got != want:
            print(f"{vid}: line list does not match the video's segments ({len(got)} vs {len(want)})"); bad += 1; continue
        a = b = 0; phrases = ph(vid)
        for i, l in enumerate(d["lines"]):
            new, why = check(l["orig"], l.get("new", ""), keeps_for(l["orig"], phrases))
            if new is None:
                bad += 1; print(f"{vid} #{i}: FAIL {why}\n   orig: {l['orig']}\n   new:  {l.get('new','')}")
            a += fk(l["orig"]); b += fk(l.get("new", ""))
        n = len(d["lines"])
        print(f"{vid:22s} ok  readability grade {a / n:.1f} -> {b / n:.1f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
