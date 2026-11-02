"""Rewrite each video's narration so a beginner can follow it on first listen.

The visuals are fixed, so every line is rewritten in the context of its whole video and of what is on
screen while it plays (the segment's scene description). Hard constraints, checked per line:
  * every number kept, in the same order
  * every phrase the animations sync to kept word for word, in the same order
  * length between 0.9x and 1.8x of the original
Lines that fail after retries fall back to the original. Results go into narration_natural.json
(original line -> spoken line), which narrate.natural() uses during rendering.

Usage: python3 video/clarify.py [--from public/videos/index.json] [ids...]
Needs DEEPSEEK_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import narrate  # noqa: E402

SYSTEM = """You rewrite the narration of a short explainer video about how language models work, so that a curious
beginner can follow it easily the first time they hear it. The visuals are fixed: each line plays while its
described visuals are on screen, so each rewritten line must still talk about what its visuals show, in order.

Make it easy to follow:
- Plain, concrete words. The first time a technical term appears in this video, explain it in a few words
  (for example: "a token, a small chunk of text", "the loss, a score for how wrong the guess was").
- One idea at a time. Short, clear sentences that still flow ("so", "which means", "here's why").
- Signpost: say what we are about to look at and why it matters, and link each line to the one before.
- Use a simple everyday comparison only when it genuinely helps, at most one every few lines.
- Warm and calm, like a good teacher sitting next to the viewer. No hype, no slang, no rhetorical tricks.

Hard rules:
- Keep every fact. Do not add claims beyond what the line says, apart from plain definitions of terms.
- Keep every number exactly as written (same digits), in the same order within its line. Do not add numbers.
- For each line, keep its "keep" phrases word for word, in the same order (capitalization may change).
- Each rewritten line may be longer than the original (up to about 1.7 times the words), never shorter than 0.9 times.
- Write for the ear: no brackets, no lists, no symbols like % or ->; write "percent" only if the original has "percent".
Return JSON: {"lines": [ ... ]} with exactly as many lines as the input, in the same order."""

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def ask(messages, max_tokens=4000, temperature=0.5, json_mode=True):
    body = {"model": "deepseek-chat", "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"], "content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]


def _at(line, p):
    m = re.search(rf"(?<!\w){re.escape(p.lower())}(?!\w)", line.lower())
    return m.start() if m else -1


def keeps_for(line, phrases):
    found = [(_at(line, p), p) for p in phrases if _at(line, p) >= 0]
    found.sort()
    # drop phrases nested inside a longer kept phrase at the same spot
    out = []
    for i, p in found:
        if not any(i >= j and i + len(p) <= j + len(q) for j, q in out):
            out.append((i, p))
    return [p for _, p in out]


def check(orig, new, keep):
    if isinstance(new, dict):
        new = new.get("line") or new.get("text") or ""
    new = narrate._decap(str(new).strip().strip('"'))
    if _NUM.findall(new) != _NUM.findall(orig):
        return None, "numbers"
    pos = -1
    for p in keep:
        j = new.lower().find(p.lower(), pos + 1)
        if j < 0:
            return None, f"missing/out-of-order {p!r}"
        pos = j
    r = len(new.split()) / max(1, len(orig.split()))
    if not 0.9 <= r <= 1.85:
        return None, f"length {r:.2f}"
    return new, "ok"


def clarify_video(v, phrases):
    segs = v["segments"]
    items = [{"i": i, "visuals": s.get("scene", ""), "line": s["orig"], "keep": keeps_for(s["orig"], phrases)} for i, s in enumerate(segs)]
    user = json.dumps({"video": v["title"], "lines": items}, ensure_ascii=False)
    out = [None] * len(segs)
    for attempt in range(3):
        try:
            got = json.loads(ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]))["lines"]
            if len(got) == len(segs):
                break
        except Exception as e:  # noqa: BLE001
            print("  retry whole script:", type(e).__name__)
    else:
        got = [s["orig"] for s in segs]
    for i, it in enumerate(items):
        new, why = check(it["line"], got[i] if i < len(got) else "", it["keep"])
        tries = 0
        while new is None and tries < 3:
            tries += 1
            ctx = {"video": v["title"], "previous_line": out[i - 1] if i else None,
                   "next_line_original": items[i + 1]["line"] if i + 1 < len(items) else None,
                   "lines": [it], "problem_last_time": why}
            try:
                g = json.loads(ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}],
                                   max_tokens=800, temperature=0.4 + 0.15 * tries))["lines"][0]
                new, why = check(it["line"], g, it["keep"])
            except Exception as e:  # noqa: BLE001
                why = type(e).__name__
        out[i] = new or it["line"]
        if new is None:
            print(f"  [{v['id']} #{i}] kept original ({why})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--from", dest="src", default=os.path.join(HERE, "..", "public", "videos", "index.json"))
    a = ap.parse_args()
    # original lines: the index may already hold spoken lines, so map back through the cache
    cache = json.load(open(narrate.NATURAL_PATH)) if os.path.exists(narrate.NATURAL_PATH) else {}
    back = {v: k for k, v in cache.items()}
    index = json.load(open(a.src))
    phrases = narrate._sync_phrases()
    for v in index:
        if a.ids and v["id"] not in a.ids:
            continue
        for s in v["segments"]:
            s["orig"] = back.get(s["text"], s["text"])
        new = clarify_video(v, phrases)
        for s, n in zip(v["segments"], new):
            cache[s["orig"]] = n
        json.dump(cache, open(narrate.NATURAL_PATH, "w"), indent=1, ensure_ascii=False)
        ow = sum(len(s["orig"].split()) for s in v["segments"]); nw = sum(len(n.split()) for n in new)
        print(f"{v['id']:22s} {len(new)} lines, words {ow} -> {nw}", flush=True)


if __name__ == "__main__":
    main()
