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
  requests.jsonl and get a plain numeric tracking code (1001, 1002, ...).
  GET /requests
  lists them for the owner. /reply also accepts an optional "tool_call"
  object with the same shape and returns its outcome as "tool_result".

Persona: the receptionist system prompt is loaded from the file named
by RECEPTIONIST_PROMPT_FILE (default: receptionist-system-prompt.txt),
so personas are swappable without touching code. The current persona
is Alireza's personal receptionist (message-taking + callback
registration only; no personal information is ever revealed).

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

Fix round 3 (2026-11-15): the user rejected the live call as too slow
and dumb. Research verdicts: Groq TTS has NO Persian (English + Arabic
Saudi Orpheus only), and every Piper fa_IR voice is medium-only, so
Piper stays the voice. The slowness was synthesis on Railway's 0.1 CPU
(~5s a reply), so stock replies are pre-synthesized at startup into an
in-memory cache and common intents get pinned replies with no LLM
round-trip, with exact pinned sentences the LLM reproduces so turns
hit the same audio cache.

Rebuild (2026-11-15): the shop scope was dropped entirely at the
user's order — the assistant is now Alireza's PERSONAL receptionist
(first name only; takes messages, registers callback requests, and
never reveals personal information). The cache/fast-path/streaming
mechanisms from fix round 3 are kept; all pinned replies, intents,
cache strings, the greeting, the Whisper domain prompt and the
tracking-code format were replaced with personal-receptionist ones.

Resilience (2026-11-15): after six fast /transcribe 502s with empty app logs, Groq failures log one line (endpoint, HTTP status, error-body excerpt) and 429/5xx gets one retry after ~2s, in both groq_transcribe and groq_chat.

Run on the VPS via systemd (see vps-setup.sh); locally with the piper venv python.
"""
import base64, io, json, os, re, sys, threading, time, wave
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
            "tracking_code": str(1001 + len(recs)),
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


# ---- TTS cache + pinned replies (fix round 3 + personal rebuild, 2026-11-15) ----
# Stock replies are synthesized once at startup (background thread; the
# port opens immediately) and served from memory. Runtime synthesis is
# also cached, so any repeated sentence is instant afterwards.
def _normalize_text(s):
    s = (s or "").strip()
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ـ", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .!؟?…،")


SYNTH_LOCK = threading.Lock()  # serialize ONNX synthesis: thread-safe, no contention on 0.1 CPU
TTS_CACHE = {}
TTS_CACHE_MAX = 200
TTS_CACHE_LOCK = threading.Lock()
WARMUP_TOTAL = 0
WARMUP_DONE = threading.Event()

# Pinned replies. These exact strings also appear in the persona file
# as the required sentences for their cases, so LLM turns reproduce
# the same text and hit the same audio cache. Keep both files in sync.
GREETING = "سلام! من منشی علیرضا هستم. چطور می‌تونم کمکتون کنم؟"
REDIRECT = "من منشی علیرضا هستم و فقط پیام می‌گیرم و درخواست تماس برگشتی ثبت می‌کنم. پیامی دارید؟"
TAKE_MESSAGE = "پیامتون رو می‌گیرم و به علیرضا می‌رسونم. نام، شماره‌ی تماس و پیامتون رو بگید."
SERVICES_REPLY = "علیرضا طراحی سایت انجام می‌ده، از جمله سایت‌های سه‌بعدی. اگه سفارش سایت دارید، پیامتون رو ثبت می‌کنم."
PRIVACY_REPLY = "نمی‌تونم اطلاعات شخصی علیرضا رو بدم. اگه پیامی دارید، می‌گیرم و بهش می‌رسونم."
UNKNOWN_REPLY = "این رو نمی‌دونم و حدس نمی‌زنم. نام و شماره‌تون رو بدید تا درخواست تماس برگشتی ثبت کنم."
CALLBACK_REPLY = "حتماً. نام، شماره‌ی تماس و موضوع رو بگید تا درخواست تماس برگشتی رو ثبت کنم."
REPEAT_REPLY = "متأسفم، درست نشنیدم. می‌شه دوباره بگید؟"


def split_sentences(text):
    # Mirrors the call page's splitter (live-call-page.html):
    # /[^.!؟?…]+[.!؟?…]*/g, trimmed. Pre-warming these exact strings is
    # what makes per-sentence /speak calls hit the cache.
    parts = re.findall(r"[^.!؟?…]+[.!؟?…]*", str(text))
    out = [p.strip() for p in parts if p.strip()]
    return out or [str(text).strip()]


def synth_wav_cached(text):
    key = _normalize_text(text)
    with TTS_CACHE_LOCK:
        hit = TTS_CACHE.get(key)
    if hit is not None:
        return hit, 0, True
    with SYNTH_LOCK:
        with TTS_CACHE_LOCK:
            hit = TTS_CACHE.get(key)
        if hit is not None:
            return hit, 0, True
        wav, ms = synth_wav(text)
        with TTS_CACHE_LOCK:
            if len(TTS_CACHE) >= TTS_CACHE_MAX:
                TTS_CACHE.pop(next(iter(TTS_CACHE)))
            TTS_CACHE[key] = wav
        return wav, ms, False


def _warm_tts_cache():
    global WARMUP_TOTAL
    texts = [GREETING, REDIRECT, TAKE_MESSAGE, SERVICES_REPLY, PRIVACY_REPLY,
             UNKNOWN_REPLY, CALLBACK_REPLY, REPEAT_REPLY]
    queue, seen = [], set()
    for t in texts:
        for s in split_sentences(t) + [t]:
            k = _normalize_text(s)
            if k and k not in seen:
                seen.add(k)
                queue.append(s)
    WARMUP_TOTAL = len(queue)
    for s in queue:
        try:
            synth_wav_cached(s)
        except Exception as e:
            print(f"tts warmup failed for {s[:30]!r}: {e}", flush=True)
    WARMUP_DONE.set()
    print(f"tts cache warmed: {len(TTS_CACHE)} entries", flush=True)


# Groq failure handling (resilience fix, 2026-11-15): a live incident
# showed six fast /transcribe 502s in a row with NOTHING in the app
# logs, so the Groq failure was unprovable retroactively. Every Groq
# failure now logs one line, and 429/5xx gets one retry after ~2s.
def _log_groq_error(endpoint, e):
    """Log one line for a failed Groq call and stash the error-body
    excerpt on the exception: e.read() consumes the body, so it must
    be read exactly once, here — the HTTP handler reuses the stash
    for the 502 JSON detail. Groq error bodies carry only error
    type/message text; keys and request bodies are never logged."""
    try:
        excerpt = e.read().decode(errors="replace")[:300]
    except Exception:
        excerpt = ""
    e.groq_body_text = excerpt
    print(f"groq {endpoint}: HTTP {e.code} {excerpt}", flush=True)


def _groq_detail(e):
    """Error-body excerpt for the 502 JSON: prefer the stash left by
    _log_groq_error; fall back to reading the body for HTTPError
    raised outside groq_transcribe/groq_chat."""
    detail = getattr(e, "groq_body_text", None)
    if detail is None:
        try:
            detail = e.read().decode(errors="replace")[:300]
        except Exception:
            detail = ""
    return detail


def _groq_retryable(code):
    return code == 429 or 500 <= code <= 599


def groq_transcribe(audio_bytes, ctype):
    boundary = "----livecall" + str(int(time.time() * 1000))
    body = b""
    def field(name, value):
        nonlocal body
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    field("model", WHISPER_MODEL)
    field("language", "fa")
    field("response_format", "json")
    # Domain prompt: Whisper hears receptionist words (پیام، تماس برگشتی،
    # علیرضا) far better when primed; without it, real phone speech
    # garbled domain words (seen live 2026-11-15: "درست نمی‌فهمه،
    # اشتباه تایپ می‌کنه").
    field("prompt", "علیرضا، پیام، تماس برگشتی، نام، شماره تماس، طراحی سایت، سایت سه‌بعدی، سفارش سایت، قرار، جلسه")
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
    # One retry on 429/5xx: Groq burst-limits are transient (seen live
    # 2026-11-15: six fast 502s, then 200 again minutes later). The
    # request is rebuilt per attempt so urlopen never reuses a request
    # object. Failures log one line (see _log_groq_error) and the
    # HTTPError still propagates, so the handler returns 502 as today.
    for attempt in range(2):
        req = urlreq.Request(f"{GROQ_BASE}/audio/transcriptions", data=body, method="POST")
        req.add_header("User-Agent", GROQ_UA)
        if not apply_groq_auth(req):
            raise RuntimeError("no Groq credential available")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with urlreq.urlopen(req, timeout=60) as r:
                return json.loads(r.read()).get("text", "").strip()
        except HTTPError as e:
            _log_groq_error("audio/transcriptions", e)
            if attempt == 0 and _groq_retryable(e.code):
                print(f"groq audio/transcriptions: retrying once after HTTP {e.code}", flush=True)
                time.sleep(2)
                continue
            raise
        except Exception as e:
            print(f"groq audio/transcriptions: {type(e).__name__}: {str(e)[:300]}", flush=True)
            raise


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
    # and non-empty. The loop caps total attempts at two: a retry fires
    # once for an empty reply OR once for a retryable Groq HTTP error
    # (429/5xx, transient burst-limits), then the fixed fallback
    # sentence (empty case) or the propagated error (HTTP case).
    for attempt in range(2):
        payload = json.dumps({"model": CHAT_MODEL, "messages": msgs,
                              "temperature": 0.3, "max_tokens": 600,
                              "reasoning_effort": "low"}).encode()
        req = urlreq.Request(f"{GROQ_BASE}/chat/completions", data=payload, method="POST")
        req.add_header("User-Agent", GROQ_UA)
        if not apply_groq_auth(req):
            raise RuntimeError("no Groq credential available")
        req.add_header("Content-Type", "application/json")
        try:
            with urlreq.urlopen(req, timeout=60) as r:
                content = json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except HTTPError as e:
            _log_groq_error("chat/completions", e)
            if attempt == 0 and _groq_retryable(e.code):
                print(f"groq chat/completions: retrying once after HTTP {e.code}", flush=True)
                time.sleep(2)
                continue
            raise
        except Exception as e:
            print(f"groq chat/completions: {type(e).__name__}: {str(e)[:300]}", flush=True)
            raise
        if content:
            return content
    return REPEAT_REPLY


_GREETINGS = {"سلام", "سلام علیکم", "درود", "وقت بخیر", "سلام وقت بخیر",
              "روز بخیر", "سلام روز بخیر", "شب بخیر", "سلام شب بخیر",
              "سلام خسته نباشید", "خسته نباشید"}
# Privacy probes: the caller asking for ALIREZA's own number/address.
# Possessive/attributed forms only — a caller offering their OWN number
# ("شماره‌م رو بدم؟") must not match.
_PRIVACY_KEYS = ("شماره‌ش", "شماره علیرضا", "شماره موبایلش", "شماره تلفنش",
                 "شماره تماسش", "موبایلش", "تلفنش", "آدرسش", "آدرس علیرضا",
                 "خونه‌ش", "منزلش")
_SERVICES_KEYS = ("چیکار", "چه کار", "شغلش", "شغل علیرضا", "کارت چیه",
                  "کار علیرضا", "سایت می‌ساز", "سایت می‌سازه")
_CALLBACK_KEYS = ("تماس برگشتی", "زنگ بزن", "زنگ بزنه", "قرار", "جلسه",
                  "سفارش سایت", "سایت می‌خوام", "سایت می‌خوایم")


def pinned_reply(text):
    """Deterministic fast path for the most common caller intents.

    Returns a pinned reply string, or None to fall through to the LLM.
    Pinned replies are pre-synthesized at startup, so these turns skip
    both the LLM round-trip and Piper synthesis.
    """
    norm = _normalize_text(text)
    if not norm:
        return None
    # A leading greeting ("سلام، علیرضا هست؟") is stripped so the real
    # intent underneath still matches; a bare greeting gets the greeting.
    m = re.match(r"^(سلام علیکم|سلام وقت بخیر|سلام روز بخیر|سلام شب بخیر|"
                 r"سلام خسته نباشید|سلام|درود|وقت بخیر)([،, ].*|$)", norm)
    if m:
        rest = m.group(2).strip(" ،,")
        if not rest:
            return GREETING
        norm = rest
    if norm in _GREETINGS:
        return GREETING
    # Privacy first: never let a probe for his number/address fall
    # through to a helpful-sounding LLM answer.
    if any(k in norm for k in _PRIVACY_KEYS):
        return PRIVACY_REPLY
    if "پیام" in norm or "پیغام" in norm:
        return TAKE_MESSAGE
    if any(k in norm for k in _CALLBACK_KEYS):
        return CALLBACK_REPLY
    if any(k in norm for k in ("کار دارم", "حرف بزنم", "صحبت کنم")):
        return TAKE_MESSAGE
    if "علیرضا" in norm and any(k in norm for k in ("هست", "کجاست", "می‌خوام")):
        return TAKE_MESSAGE
    if any(k in norm for k in _SERVICES_KEYS):
        return SERVICES_REPLY
    return None


def get_reply(text, history):
    pinned = pinned_reply(text)
    if pinned is not None:
        return pinned, True
    return groq_chat(text, history), False


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
                             "persona_file": PROMPT_FILE,
                             "tts_cache": {"entries": len(TTS_CACHE),
                                           "warmup_done": WARMUP_DONE.is_set(),
                                           "warmup_total": WARMUP_TOTAL}})
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
            wav, ms, cached = synth_wav_cached(text)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.send_header("X-Synth-Ms", str(ms))
            self.send_header("X-Cache", "hit" if cached else "miss")
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
                                        "detail": _groq_detail(e)})
            except Exception as e:
                return self._json(502, {"error": "groq_error", "detail": str(e)[:300]})
            return self._json(200, {"text": text,
                                    "ms": round((time.time() - t0) * 1000)})

        # Text-only reply: the call page shows the text immediately and
        # streams speech sentence-by-sentence via /speak, so the first
        # sound arrives after one short sentence of synthesis instead of
        # after the whole reply (Piper on Railway's 0.1 CPU took ~5s for
        # a full reply — the "خیلی کنده" complaint, 2026-11-15).
        if self.path == "/reply_text":
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
                reply, fast = get_reply(text, history)
            except HTTPError as e:
                return self._json(502, {"error": "groq_error", "status": e.code,
                                        "detail": _groq_detail(e)})
            except Exception as e:
                return self._json(502, {"error": "reply_failed", "detail": str(e)[:300]})
            resp = {"reply": reply, "fast_path": fast,
                    "ms": round((time.time() - t0) * 1000)}
            if isinstance(tool_call, dict):
                _code, resp["tool_result"] = store_request(tool_call)
            self._json(200, resp)
            return

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
                reply, fast = get_reply(text, history)
                wav, synth_ms, cached = synth_wav_cached(reply)
            except HTTPError as e:
                return self._json(502, {"error": "groq_error", "status": e.code,
                                        "detail": _groq_detail(e)})
            except Exception as e:
                return self._json(502, {"error": "reply_failed", "detail": str(e)[:300]})
            resp = {"reply": reply,
                    "audio_b64": base64.b64encode(wav).decode(),
                    "audio_mime": "audio/wav",
                    "ms": round((time.time() - t0) * 1000),
                    "synth_ms": synth_ms, "fast_path": fast,
                    "cache": "hit" if cached else "miss"}
            if isinstance(tool_call, dict):
                # Same code-enforced store as POST /tools/request; the
                # outcome rides back so the page can show the tracking code.
                _code, resp["tool_result"] = store_request(tool_call)
            self._json(200, resp)
            return

        self._json(404, {"error": "not found"})


if __name__ == "__main__":
    # Pre-synthesize stock replies in the background: the port opens
    # immediately, and the cache fills while the first calls come in.
    threading.Thread(target=_warm_tts_cache, daemon=True).start()
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
