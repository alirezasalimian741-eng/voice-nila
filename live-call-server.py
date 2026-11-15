"""Live Persian voice-call server: Groq (hear + brain) + Piper Amir (voice).

One server for the real call page (space:dstyr-swty). Browser flow:
  mic audio (webm/wav) -> POST /transcribe -> Persian text
  text + history        -> POST /reply      -> {reply, wav audio}
  (or POST /speak for direct TTS)
The call page itself is served by this server at GET / from
live-call-page.html, so mic -> Groq -> Piper runs end to end on one
origin (no CORS workaround needed). Without a Groq key the page shows
exactly that and disables the mic button; /speak still works.

Tools layer ("software decides, model requests" — BUILD-STANDARD.md):
  POST /tools/request {type: booking|callback, name, phone, topic,
  when_text, idempotency_key, confirmed} registers an order/callback.
  Rules are enforced HERE, in code, not in the prompt: confirmed is not
  true -> 422 confirmation_required and nothing is stored; an already
  seen idempotency_key returns the ORIGINAL record with duplicate:true
  instead of storing a second one. Records append one JSON line each to
  requests.jsonl and get a tracking code (NV-0001, ...). GET /requests
  lists them for the owner. /reply also accepts an optional "tool_call"
  object with the same shape and returns its outcome as "tool_result".

Persona: the receptionist system prompt is loaded from the file named
by RECEPTIONIST_PROMPT_FILE (default: receptionist-system-prompt.txt,
extracted from receptionist-nila-brain.md §4), so personas are swappable
without touching code.

Groq auth has two sources, in order: the GROQ_API_KEY environment
variable (the VPS/host path), or the stored Muse connection custom.groq
via an authd surrogate (this VM only). The key itself is never read
from chat and never written to disk. With neither, /transcribe and
/reply answer 503 {"error": "groq_key_missing"}; /health and /speak
still work (Piper is local).

Models (checked 2026-11-05 against Groq rate-limit docs):
  hear:  whisper-large-v3-turbo — free 20 req/min, 2,000/day, 7,200 audio-sec/hour
  brain: openai/gpt-oss-120b   — free 30 req/min, 1,000/day, 200,000 tokens/day
  voice: Piper fa_IR-amir-medium, tuned (length 1.12, noise 0.9) — measured
         0.36s for "سلام علیرضا، چطوری؟" on this VM, 2026-11-05.
Expected reply latency: Whisper ~1-2s + LLM ~1s + Piper ~0.4s = ~2-4s.
Groq endpoints are its OpenAI-compatible ones. The Groq path was
tested LIVE 2026-11-15 (key connected via custom.groq): /transcribe
826ms, /reply ~2.6-2.8s end to end. Two live lessons are baked in:
(1) every Groq request must carry a browser-like User-Agent or
Cloudflare returns 1010 before auth is checked; (2) gpt-oss-120b
needs reasoning_effort=low + max_tokens>=600 or it can return empty
content after spending the budget on reasoning.

Run on the VPS via systemd (see vps-setup.sh); locally with the piper venv python.
"""
import base64, io, json, os, sys, threading, time, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlreq
from urllib.error import HTTPError

pass  # VPS: piper-tts is installed in /opt/voice/venv
from piper import PiperVoice, SynthesisConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("PIPER_MODEL", os.path.join(BASE_DIR, "models/fa_IR-amir-medium.onnx"))
GROQ_BASE = "https://api.groq.com/openai/v1"
WHISPER_MODEL = "whisper-large-v3-turbo"
CHAT_MODEL = "openai/gpt-oss-120b"
PORT = int(os.environ.get("PORT", "8766"))
HOST = os.environ.get("HOST", "127.0.0.1")  # systemd unit sets HOST=0.0.0.0 on the VPS

PROMPT_FILE = os.environ.get(
    "RECEPTIONIST_PROMPT_FILE",
    os.path.join(BASE_DIR, "receptionist-system-prompt.txt"))


def load_system_prompt():
    try:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return text
    except OSError:
        pass
    # Fallback if the prompt file is missing: stay a cautious receptionist.
    return ("تو منشی تلفنی فارسی هستی؛ کوتاه و طبیعی حرف بزن. فقط از روی "
            "اطلاعاتی که داری جواب بده و اگر چیزی را نمی‌دانی حدس نزن؛ "
            "نام و شماره‌ی تماس‌گیرنده را برای تماس برگشتی بگیر.")


SYSTEM_PROMPT = load_system_prompt()

voice = PiperVoice.load(MODEL)
CFG = SynthesisConfig(length_scale=1.12, noise_scale=0.9, noise_w_scale=0.9)
print("piper model loaded", flush=True)

# ---- tools layer: the software decides, the model only requests ----
REQUESTS_FILE = os.environ.get(
    "REQUESTS_FILE", os.path.join(BASE_DIR, "requests.jsonl"))
REQ_LOCK = threading.Lock()


def load_requests():
    recs = []
    try:
        with open(REQUESTS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    except OSError:
        pass
    return recs


def store_request(data):
    """Register one booking/callback. Returns (http_code, payload).

    Code-enforced rules (BUILD-STANDARD): caller confirmation is a hard
    gate — confirmed is not True -> 422 and nothing is stored. An
    idempotency_key that was already stored returns the ORIGINAL record
    with duplicate:true; a request is never stored twice.
    """
    if not isinstance(data, dict) or data.get("confirmed") is not True:
        return 422, {"error": "confirmation_required"}
    if data.get("type") not in ("booking", "callback"):
        return 400, {"error": "bad type"}
    idem = (data.get("idempotency_key") or "").strip()
    if not idem:
        return 400, {"error": "idempotency_key_required"}
    with REQ_LOCK:
        recs = load_requests()
        for r in recs:
            if r.get("idempotency_key") == idem:
                out = dict(r)
                out["duplicate"] = True
                return 200, out
        rec = {
            "tracking_code": "NV-" + str(len(recs) + 1).zfill(4),
            "type": data["type"],
            "name": (data.get("name") or "").strip(),
            "phone": (data.get("phone") or "").strip(),
            "topic": (data.get("topic") or "").strip(),
            "when_text": (data.get("when_text") or "").strip(),
            "idempotency_key": idem,
            "confirmed": True,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(REQUESTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return 201, rec


def groq_key():
    return os.environ.get("GROQ_API_KEY", "").strip() or None


# Auth for Groq calls. Two sources, in order:
#   1. GROQ_API_KEY env var (the VPS/host path — the key lives on the host).
#   2. The stored Muse connection custom.groq, via an authd surrogate
#      (this VM only; Sentinel swaps the surrogate for the real key on
#      approved egress to api.groq.com). The key itself is never readable
#      here and never written to disk.
_surrogate_cache = {"ok": None, "at": 0.0}


def _surrogate_auth(req):
    try:
        _bin = "/opt/muse/skills/custom-connector-setup/bin"
        if _bin not in sys.path:
            sys.path.insert(0, _bin)
        from dynamic_credentials import add_surrogate_to_request
        add_surrogate_to_request(req, "custom.groq", allowed_hosts=["api.groq.com"])
        return True
    except Exception:
        return False


def _surrogate_probe():
    try:
        _bin = "/opt/muse/skills/custom-connector-setup/bin"
        if _bin not in sys.path:
            sys.path.insert(0, _bin)
        from dynamic_credentials import dynamic_credential_entry
        dynamic_credential_entry("custom.groq", timeout=2.0)
        return True
    except Exception:
        return False


def apply_groq_auth(req):
    key = groq_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
        return True
    return _surrogate_auth(req)


def groq_available():
    if groq_key():
        return True
    now = time.time()
    if _surrogate_cache["ok"] is not None and now - _surrogate_cache["at"] < 60:
        return _surrogate_cache["ok"]
    ok = _surrogate_probe()
    _surrogate_cache.update(ok=ok, at=now)
    return ok


# Groq sits behind Cloudflare: the default Python-urllib User-Agent gets a
# 1010 block at the edge before auth is ever checked (seen live 2026-11-15).
# Every Groq request must carry a browser-like User-Agent.
GROQ_UA = "Mozilla/5.0 (X11; Linux x86_64) Muse-voice-server/1.0"


def synth_wav(text):
    t0 = time.time()
    chunks = list(voice.synthesize(text, syn_config=CFG))
    if not chunks:
        # Piper produced nothing (e.g. unspeakable input): fall back to a
        # fixed safe sentence instead of crashing on chunks[0].
        chunks = list(voice.synthesize(
            "متأسفم، صدا آماده نشد. لطفاً دوباره بگید.", syn_config=CFG))
    if not chunks:
        raise RuntimeError("piper produced no audio")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(chunks[0].sample_rate)
        for c in chunks:
            w.writeframes(c.audio_int16_array.tobytes())
    return buf.getvalue(), round((time.time() - t0) * 1000)


def groq_transcribe(audio_bytes, ctype):
    boundary = "----livecall" + str(int(time.time() * 1000))
    body = b""
    def field(name, value):
        nonlocal body
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    field("model", WHISPER_MODEL)
    field("language", "fa")
    field("response_format", "json")
    # Whisper keys the decoder off the filename extension, so it MUST match
    # the actual bytes: iOS Safari records audio/mp4 (NOT webm). Sending
    # mp4 bytes named .wav made Groq fail the transcription (seen live
    # 2026-11-15: the call page heard nothing on his iPhone).
    ct = (ctype or "").lower()
    if "mp4" in ct or "m4a" in ct:
        ext = "mp4"
    elif "webm" in ct:
        ext = "webm"
    elif "ogg" in ct:
        ext = "ogg"
    elif "wav" in ct or "x-wav" in ct:
        ext = "wav"
    else:
        ext = "mp4"  # our only client is his iPhone (Safari records mp4)
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"mic.{ext}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
    body += audio_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urlreq.Request(f"{GROQ_BASE}/audio/transcriptions", data=body, method="POST")
    req.add_header("User-Agent", GROQ_UA)
    if not apply_groq_auth(req):
        raise RuntimeError("no Groq credential available")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urlreq.urlopen(req, timeout=60) as r:
        return json.loads(r.read()).get("text", "").strip()


def groq_chat(text, history):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-10:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": text})
    # gpt-oss-120b is a reasoning model: with default reasoning effort it
    # can spend the whole token budget thinking and return EMPTY content
    # (seen live 2026-11-15 on a longer order message at max_tokens=300).
    # Low reasoning effort + a 600-token floor keeps replies short, fast
    # and non-empty. One retry, then a fixed fallback sentence.
    for attempt in range(2):
        payload = json.dumps({"model": CHAT_MODEL, "messages": msgs,
                              "temperature": 0.7, "max_tokens": 600,
                              "reasoning_effort": "low"}).encode()
        req = urlreq.Request(f"{GROQ_BASE}/chat/completions", data=payload, method="POST")
        req.add_header("User-Agent", GROQ_UA)
        if not apply_groq_auth(req):
            raise RuntimeError("no Groq credential available")
        req.add_header("Content-Type", "application/json")
        with urlreq.urlopen(req, timeout=60) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"].strip()
        if content:
            return content
    return "متأسفم، جوابتون رو درست نگرفتم. می‌شه دوباره بگید؟"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        if self.path in ("/", "/call"):
            try:
                with open(os.path.join(BASE_DIR, "live-call-page.html"),
                          "rb") as f:
                    page = f.read()
            except OSError:
                return self._json(404, {"error": "call page missing"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            # Never cache the call page: after a redeploy his phone must
            # get the new version, not a stale copy.
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path == "/health":
            self._json(200, {"ok": True, "voice": "fa_IR-amir-medium",
                             "groq_configured": groq_available(),
                             "whisper": WHISPER_MODEL, "chat": CHAT_MODEL,
                             "persona_file": PROMPT_FILE})
        elif self.path == "/requests":
            self._json(200, load_requests())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/tools/request":
            try:
                data = json.loads(self._read())
            except Exception:
                return self._json(400, {"error": "bad json"})
            code, out = store_request(data)
            return self._json(code, out)

        if self.path == "/speak":
            try:
                text = json.loads(self._read()).get("text", "").strip()
            except Exception:
                return self._json(400, {"error": "bad json"})
            if not text:
                return self._json(400, {"error": "empty text"})
            wav, ms = synth_wav(text)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.send_header("X-Synth-Ms", str(ms))
            self.end_headers()
            self.wfile.write(wav)
            return

        if self.path == "/transcribe":
            if not groq_available():
                return self._json(503, {"error": "groq_key_missing",
                                        "hint": "no GROQ_API_KEY env var and no stored custom.groq connection"})
            ctype = self.headers.get("Content-Type") or "audio/webm"
            t0 = time.time()
            try:
                text = groq_transcribe(self._read(), ctype)
            except HTTPError as e:
                return self._json(502, {"error": "groq_error", "status": e.code,
                                        "detail": e.read().decode(errors="replace")[:300]})
            except Exception as e:
                return self._json(502, {"error": "groq_error", "detail": str(e)[:300]})
            return self._json(200, {"text": text,
                                    "ms": round((time.time() - t0) * 1000)})

        if self.path == "/reply":
            if not groq_available():
                return self._json(503, {"error": "groq_key_missing",
                                        "hint": "no GROQ_API_KEY env var and no stored custom.groq connection"})
            try:
                data = json.loads(self._read())
                text = (data.get("text") or "").strip()
                history = data.get("history") or []
                tool_call = data.get("tool_call")
            except Exception:
                return self._json(400, {"error": "bad json"})
            if not text:
                return self._json(400, {"error": "empty text"})
            t0 = time.time()
            try:
                reply = groq_chat(text, history)
                wav, synth_ms = synth_wav(reply)
            except HTTPError as e:
                return self._json(502, {"error": "groq_error", "status": e.code,
                                        "detail": e.read().decode(errors="replace")[:300]})
            except Exception as e:
                return self._json(502, {"error": "reply_failed", "detail": str(e)[:300]})
            resp = {"reply": reply,
                    "audio_b64": base64.b64encode(wav).decode(),
                    "audio_mime": "audio/wav",
                    "ms": round((time.time() - t0) * 1000),
                    "synth_ms": synth_ms}
            if isinstance(tool_call, dict):
                # Same code-enforced store as POST /tools/request; the
                # outcome rides back so the page can show the tracking code.
                _code, resp["tool_result"] = store_request(tool_call)
            self._json(200, resp)
            return

        self._json(404, {"error": "not found"})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
