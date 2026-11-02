"""Narration audio for the explainer videos: speech synthesis, gentle mastering, and word alignment.

Engines (env VIDEO_TTS):
  aura       Deepgram Aura-2 on Cloudflare Workers AI (needs CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN)
  kokoro     Kokoro-82M, run locally by video/tts_local.py inside the Python given by env TTS_PYTHON
  chatterbox Chatterbox, same mechanism as kokoro
Voice: env VIDEO_VOICE (engine-specific name).

What makes it sound like a person rather than a machine:
  * no speed-up and no pause squashing: the engine's own phrasing and breaths are kept;
    only pauses longer than ~1 s are shortened, and leading/trailing silence is trimmed
  * lossless 24 kHz audio from the engine (not low-bitrate MP3)
  * light mastering: high-pass, gentle compression, consistent loudness
  * text is rewritten for the ear (spoken() below), captions keep the original text
Word timestamps come from Whisper (Workers AI) so animations can land on the exact word.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
ENGINE = os.environ.get("VIDEO_TTS", "aura")
VOICE = os.environ.get("VIDEO_VOICE", "")
PIPE = "v3"
LOCAL_PORT = int(os.environ.get("TTS_PORT", "5899"))

DEFAULT_VOICE = {"aura": "odysseus", "kokoro": "am_michael", "chatterbox": "default"}


def voice() -> str:
    return VOICE or DEFAULT_VOICE[ENGINE]


# ---- text for the ear ------------------------------------------------------------
_SAY = [
    (r"\bTinyGPT\b", "Tiny GPT"), (r"\bAdamW\b", "Adam W"), (r"\bLayerNorm\b", "layer norm"),
    (r"\bRMSNorm\b", "R M S norm"), (r"\bSwiGLU\b", "swiggloo"), (r"\bGELU\b", "jell-oo"),
    (r"\bKV\b", "K V"), (r"\bMLPs\b", "M L Ps"), (r"\bMLP\b", "M L P"), (r"\bSGD\b", "S G D"),
    (r"\bPCA\b", "P C A"), (r"\bRoPE\b", "rope"), (r"\bBPE\b", "B P E"), (r"\bRLHF\b", "R L H F"),
    (r"\bDPO\b", "D P O"), (r"\bSFT\b", "S F T"), (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
    (r"\bvs\.?", "versus"), (r"≈", " about "), (r"×", " times "), (r"→", " to "), (r"−", "minus "),
    (r"\b1e-4\b", "one in ten thousand"), (r"\b1e-3\b", "one in a thousand"), (r"\b3e-3\b", "three in a thousand"),
    (r"\bln\b", "natural log"), (r"\s*—\s*", ", "), (r"\s*–\s*", ", "),
]


def spoken(text: str) -> str:
    s = text
    for pat, rep in _SAY:
        s = re.sub(pat, rep, s)
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)          # 4,096 -> 4096 (engines read plain numbers well)
    # 5-6 digit numbers come out digit by digit ("one seven zero five six zero"): say 170560 as "170 thousand 560"
    s = re.sub(r"(?<![\d.])(\d{2,3})(\d{3})(?!\d|\.\d)",
               lambda m: f"{int(m[1])} thousand" + (f" {int(m[2])}" if int(m[2]) else ""), s)
    s = re.sub(r"(\d{4})-(?=[A-Za-z])", r"\1 ", s)  # "3000-step" was read as "3-0-0-0 step"
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---- synthesis -------------------------------------------------------------------------
def _cf(model: str, payload: dict, raw: bool) -> bytes | dict:
    url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/run/{model}"
    body = json.dumps(payload).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, data=body, method="POST", headers={
                "Authorization": "Bearer " + os.environ["CLOUDFLARE_API_TOKEN"], "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            return data if raw else json.loads(data)["result"]
        except Exception as e:  # noqa: BLE001
            print(f"[narrate] {model} retry {attempt + 1}: {type(e).__name__}")
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"{model} failed")


_server = None


def _local(text: str, out: str):
    """Ask the local engine server (started on demand in the TTS_PYTHON environment) to write `out`."""
    global _server
    url = f"http://127.0.0.1:{LOCAL_PORT}/"
    body = json.dumps({"text": text, "voice": voice(), "out": out}).encode()
    for attempt in range(240):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"), timeout=1800) as r:
                r.read(); return
        except (ConnectionRefusedError, urllib.error.URLError):
            if _server is None:
                py = os.environ.get("TTS_PYTHON")
                if not py:
                    raise RuntimeError("set TTS_PYTHON to a Python with the local TTS engine installed")
                _server = subprocess.Popen([py, os.path.join(HERE, "tts_local.py"), ENGINE, str(LOCAL_PORT)])
            time.sleep(2)
    raise RuntimeError("local TTS server did not start")


def synth(text: str, raw_path: str):
    if ENGINE == "aura":
        data = _cf("@cf/deepgram/aura-2-en", {"text": text, "speaker": voice(), "encoding": "linear16",
                                               "container": "wav", "sample_rate": 24000}, raw=True)
        if len(data) < 2000:
            raise RuntimeError("tiny TTS response")
        open(raw_path, "wb").write(data)
    else:
        _local(text, raw_path)


# Keep natural phrasing: trim the ends, shorten only pauses longer than ~1 s, then master lightly.
_MASTER = ("silenceremove=start_periods=1:start_threshold=-55dB,areverse,"
           "silenceremove=start_periods=1:start_threshold=-55dB,areverse,"
           "silenceremove=stop_periods=-1:stop_duration=1.0:stop_threshold=-50dB:stop_silence=0.7,"
           "highpass=f=70,acompressor=threshold=-21dB:ratio=2.2:attack=15:release=180:makeup=1.5,"
           "alimiter=limit=0.95")


def _dur(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def tts(text: str) -> tuple[str, float]:
    """(wav_path, duration) for a narration line; synthesized, mastered and cached."""
    os.makedirs(CACHE, exist_ok=True)
    say = spoken(text)
    key = hashlib.sha1(f"{ENGINE}|{voice()}|{say}".encode()).hexdigest()[:20]
    wav = os.path.join(CACHE, f"{key}_{PIPE}.wav")
    if not os.path.exists(wav):
        raw = os.path.join(CACHE, f"{key}_raw.wav")
        if not os.path.exists(raw):
            synth(say, raw)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", _MASTER, "-ar", "48000", "-ac", "1", wav], check=True)
    return wav, _dur(wav)


def words(wav: str) -> list[dict]:
    """Word timestamps for a narration wav (Whisper large-v3-turbo on Workers AI), cached."""
    path = wav[:-4] + ".words.json"
    if os.path.exists(path):
        return json.load(open(path))
    try:
        r = _cf("@cf/openai/whisper-large-v3-turbo",
                {"audio": base64.b64encode(open(wav, "rb").read()).decode(), "language": "en", "word_timestamps": True}, raw=False)
        ws = [{"w": w["word"].strip(), "s": w["start"], "e": w["end"]} for seg in r.get("segments", []) for w in seg.get("words", [])]
    except Exception:  # noqa: BLE001
        ws = []
    json.dump(ws, open(path, "w"))
    return ws


_norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
_NUMWORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
             "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve"}


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    if _NUMWORDS.get(a) == b or _NUMWORDS.get(b) == a:
        return True
    short, long_ = sorted((a, b), key=len)   # allow plurals / possessives ("token" ~ "tokens"), not "to" ~ "tom"
    return len(short) >= 4 and long_.startswith(short) and len(long_) - len(short) <= 2


def phrase_time(ws: list[dict], phrase: str, after: float = 0.0) -> float | None:
    """Start time of `phrase` in the aligned words, searching only from `after` seconds on (sync is sequential)."""
    target = [_norm(t) for t in spoken(phrase).split() if _norm(t)]
    if not ws or not target:
        return None
    toks = [_norm(w["w"]) for w in ws]
    k = min(3, len(target))
    for need in (k, 1):
        for i in range(len(toks)):
            if ws[i]["e"] < after:
                continue
            if all(i + j < len(toks) and _same(toks[i + j], target[j]) for j in range(need)):
                return ws[i]["s"]
    return None


# ---- natural phrasing ---------------------------------------------------------------
# The scene scripts were written in a clipped "punchline" style that sounds robotic when synthesized.
# natural() rewrites each line into flowing spoken English with an LLM, under hard constraints
# (every number and every phrase the animations sync to is kept verbatim), and caches the result in
# narration_natural.json so builds are reproducible and the text can be reviewed or hand-edited.
NATURAL_PATH = os.path.join(HERE, "narration_natural.json")
_natural: dict | None = None


def _sync_phrases(scene_file: str | None = None) -> list[str]:
    """Literal phrases passed to self.sync()/t_of() — in one scene file if given, else everywhere."""
    out = set()
    files = [os.path.join(HERE, "scenes", scene_file)] if scene_file else \
        [os.path.join(r, f) for r in (os.path.join(HERE, "scenes"), HERE) for f in os.listdir(r) if f.endswith(".py")]
    for path in files:
        src = open(path).read()
        out.update(re.findall(r'(?:self\.sync|t_of)\(\s*"([^"{}]+)"', src))
        out.update(re.findall(r"(?:self\.sync|t_of)\(\s*'([^'{}]+)'", src))
    return sorted(out, key=len, reverse=True)


def scene_file(video_id: str) -> str | None:
    """Which scenes/*.py renders a video (BUILTIN list or a VIDEO dict in the file)."""
    import ast
    src = open(os.path.join(HERE, "build.py")).read()
    for m in re.finditer(r'id="([^"]+)", file="([^"]+)"', src):
        if m.group(1) == video_id:
            return m.group(2)
    for f in os.listdir(os.path.join(HERE, "scenes")):
        if f.endswith(".py") and re.search(rf'VIDEO\s*=\s*dict\([^)]*id\s*=\s*"{re.escape(video_id)}"', open(os.path.join(HERE, "scenes", f)).read(), re.S):
            return f
    return None


_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_PROMPT = """You rewrite one line of narration for an explainer video about how language models work.
A warm, curious human narrator (think 3Blue1Brown) will read it aloud. Make it flow like natural speech:
join choppy fragments into full sentences, use contractions and light connecting words ("so", "and", "which means"),
vary the rhythm, keep it clear and friendly. Rules:
- Keep the meaning and every fact exactly. Do not add new claims, examples or jokes.
- Keep every number exactly as written (same digits, same order), e.g. "87 percent" stays "87 percent".
- Keep these phrases word for word if they appear in the line (capitalization may change to fit the sentence): {keep}
- Keep technical terms and token names (like "red", "the", TinyGPT, softmax) unchanged.
- About the same length: between 90% and 130% of the original word count.
- Output only the rewritten line, no quotes."""


_COMMON = "The|Nobody|It|This|That|And|But|So|Now|Each|Every|Then|Here|Its|They|We|You|A|An|In|On|At|When|If|With|One|Two|All|What|Which|There|Same|Only"


def _decap(s: str) -> str:
    """A kept phrase can arrive capitalized mid-sentence ("..., The winner goes"); fix the casing."""
    return re.sub(rf"([,;:—]\s+|\b(?:and|but|so|that|which|though|because|while)\s+)({_COMMON})\b", lambda m: m.group(1) + m.group(2).lower(), s)


def natural(text: str) -> str:
    global _natural
    if os.environ.get("VIDEO_NATURAL", "1") == "0":
        return text
    if _natural is None:
        _natural = json.load(open(NATURAL_PATH)) if os.path.exists(NATURAL_PATH) else {}
    if text in _natural:
        return _natural[text]
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return text
    keep = [p for p in _sync_phrases() if re.search(rf"(?<!\w){re.escape(p.lower())}(?!\w)", text.lower())]
    best = text
    for attempt in range(4):
        body = json.dumps({"model": "deepseek-chat", "temperature": 0.4 + 0.15 * attempt, "max_tokens": 600, "messages": [
            {"role": "system", "content": _PROMPT.format(keep="; ".join(f'"{p}"' for p in keep) or "(none)")},
            {"role": "user", "content": text}]}).encode()
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body, method="POST",
                                         headers={"Authorization": "Bearer " + key, "content-type": "application/json"})
            out = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"].strip().strip('"')
        except Exception as e:  # noqa: BLE001
            print(f"[natural] retry {attempt + 1}: {type(e).__name__}"); time.sleep(3); continue
        out = _decap(out)
        ok_nums = _NUM.findall(out) == _NUM.findall(text)
        ok_keep = all(p.lower() in out.lower() for p in keep)
        ratio = len(out.split()) / max(1, len(text.split()))
        if ok_nums and ok_keep and 0.85 <= ratio <= 1.4:
            best = out
            break
        print(f"[natural] rejected (nums {ok_nums}, keep {ok_keep}, ratio {ratio:.2f}): {out[:70]!r}")
    _natural[text] = best
    import fcntl  # merge-and-write under a lock: several renders may run in parallel
    with open(NATURAL_PATH + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.load(open(NATURAL_PATH)) if os.path.exists(NATURAL_PATH) else {}
        cur.update({text: best}); _natural.update(cur)
        json.dump(cur, open(NATURAL_PATH, "w"), indent=1, ensure_ascii=False)
    return best
