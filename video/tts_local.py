"""Tiny local TTS server used by narrate.py for engines that run on this machine.

Run inside a Python environment that has the engine installed:
    kokoro:     pip install kokoro soundfile 'misaki[en]'   (+ apt install espeak-ng)
    chatterbox: pip install chatterbox-tts
Usage: python tts_local.py <kokoro|chatterbox> <port>.  POST {"text", "voice", "out"} -> writes a wav to `out`.
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

engine, port = sys.argv[1], int(sys.argv[2])
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("TTS_THREADS", "4")))

if engine == "kokoro":
    import numpy as np, soundfile as sf
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")

    def speak(text, voice, out):
        audio = [a for _, _, a in pipe(text, voice=voice, speed=1.0)]
        sf.write(out, np.concatenate(audio), 24000)
else:
    import perth
    perth.PerthImplicitWatermarker = perth.DummyWatermarker
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device="cpu")

    def speak(text, voice, out):
        ta.save(out, model.generate(text, exaggeration=0.5, cfg_weight=0.5), model.sr)


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        speak(req["text"], req["voice"], req["out"])
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", port), H).serve_forever()
