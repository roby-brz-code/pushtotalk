#!/usr/bin/env python3
"""
Push-to-Talk — Hold a hotkey, speak, and have your speech typed into any focused text field.

Includes a web UI at http://localhost:8528 for microphone selection,
real-time status, and transcription history.

Features:
- Cloud transcription via Groq Whisper API (fast, accurate) with local fallback
- LLM cleanup via Claude Haiku (removes fillers, adds punctuation)
- Voice commands (new line, period, scratch that, etc.)
- Custom dictionary for technical terms
"""

import datetime
import io
import json
import logging
import os
import pathlib
import platform
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

import numpy as np
import pyautogui
import pyperclip
import pystray
import sounddevice as sd
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image, ImageDraw
from pynput import keyboard as pynput_keyboard

# ─── Configuration ──────────────────────────────────────────────────────────────

APP_NAME = "Push to Talk"
HOTKEY = pynput_keyboard.Key.alt_l  # macOS Option key; change as desired
WHISPER_MODEL = "base.en"   # tiny.en | base.en | small.en | medium.en | large
SAMPLE_RATE = 16000         # 16 kHz mono
LANGUAGE = "en"             # Set to None for auto-detect
LOG_FILE = "transcription_log.txt"  # Set to None to disable logging
WEB_PORT = 8528             # Web UI port

# Cloud APIs (set via environment variables; leave unset for offline-only mode)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Custom dictionary file path
DICTIONARY_FILE = pathlib.Path.home() / ".pushtotalk" / "dictionary.txt"

# ─── Globals ────────────────────────────────────────────────────────────────────

model = None
recording = False
audio_frames = []
stream = None
tray_icon = None
key_listener = None
selected_device = None      # None = system default; set to device index to override
app_state = "idle"          # idle | recording | transcribing
transcription_history = []  # [{timestamp, text, duration_s}]
sse_clients = []            # [queue.Queue] for SSE connections
record_start_time = None    # When recording started
custom_dictionary = []      # List of custom terms from dictionary file
last_typed_text = None      # For "scratch that" voice command

# ─── Flask app ──────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)

# Suppress Flask/werkzeug request logs
werkzeug_log = logging.getLogger("werkzeug")
werkzeug_log.setLevel(logging.ERROR)


@flask_app.route("/")
def web_index():
    return render_template("index.html")


@flask_app.route("/api/status")
def api_status():
    device_name = "System Default"
    if selected_device is not None:
        try:
            device_name = sd.query_devices(selected_device)["name"]
        except Exception:
            device_name = f"Device {selected_device}"

    engine = "Groq (whisper-large-v3-turbo)" if GROQ_API_KEY else f"Local ({WHISPER_MODEL})"
    cleanup = "Claude Haiku" if ANTHROPIC_API_KEY else "Off"

    return jsonify({
        "state": app_state,
        "model": WHISPER_MODEL,
        "device": device_name,
        "device_index": selected_device,
        "hotkey": str(HOTKEY),
        "language": LANGUAGE,
        "engine": engine,
        "cleanup": cleanup,
        "dictionary_count": len(custom_dictionary),
    })


@flask_app.route("/api/devices")
def api_devices():
    return jsonify(get_input_devices())


@flask_app.route("/api/device", methods=["POST"])
def api_set_device():
    data = request.get_json(force=True)
    idx = data.get("index")  # None for system default
    set_input_device(idx)
    return jsonify({"ok": True, "device_index": selected_device})


@flask_app.route("/api/history")
def api_history():
    return jsonify(transcription_history[-50:])  # Last 50 entries


@flask_app.route("/events")
def sse_stream():
    """Server-Sent Events endpoint for real-time updates."""
    def generate():
        q = queue.Queue()
        sse_clients.append(q)
        try:
            # Send current state immediately
            yield f"event: state\ndata: {json.dumps({'state': app_state})}\n\n"
            while True:
                try:
                    message = q.get(timeout=30)
                    yield message
                except queue.Empty:
                    # Send keepalive comment to prevent timeout
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            try:
                sse_clients.remove(q)
            except ValueError:
                pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Utility helpers ────────────────────────────────────────────────────────────


def log(emoji: str, message: str) -> None:
    """Print a timestamped status message."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {emoji} {message}")


def log_transcription(text: str) -> None:
    """Append transcription to the log file with a timestamp."""
    if LOG_FILE is None:
        return
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")
    except OSError as exc:
        log("❌", f"Failed to write log: {exc}")


def broadcast_event(event_type: str, data: dict) -> None:
    """Push an SSE event to all connected web clients."""
    message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for q in sse_clients:
        try:
            q.put_nowait(message)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            sse_clients.remove(q)
        except ValueError:
            pass


def set_state(new_state: str) -> None:
    """Update app state and broadcast to all web UI clients."""
    global app_state
    app_state = new_state
    broadcast_event("state", {"state": new_state})


def beep(frequency: int = 800, duration_ms: int = 80) -> None:
    """Play a short sine-wave beep (non-blocking, best-effort)."""
    try:
        t = np.linspace(0, duration_ms / 1000, int(SAMPLE_RATE * duration_ms / 1000), dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * frequency * t)
        sd.play(tone, samplerate=SAMPLE_RATE)
    except Exception:
        pass


def send_notification(title: str, message: str) -> None:
    """Send a desktop notification (best-effort, cross-platform)."""
    system = platform.system()
    try:
        if system == "Darwin":
            os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
        elif system == "Linux":
            os.system(f'notify-send "{title}" "{message}"')
        elif system == "Windows":
            ps = (
                f'[Windows.UI.Notifications.ToastNotificationManager,'
                f' Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; '
                f'$t = [Windows.UI.Notifications.ToastNotificationManager]'
                f'::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); '
                f'$t.GetElementsByTagName("text")[0].AppendChild($t.CreateTextNode("{title}")) | Out-Null; '
                f'$t.GetElementsByTagName("text")[1].AppendChild($t.CreateTextNode("{message}")) | Out-Null; '
                f'[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("PushToTalk")'
                f'.Show([Windows.UI.Notifications.ToastNotification]::new($t))'
            )
            os.system(f'powershell -Command \'{ps}\'')
    except Exception:
        pass


# ─── Microphone icon generation ─────────────────────────────────────────────────


def _draw_mic(draw, cx, cy, size, mic_color="white"):
    """Draw a microphone shape onto an ImageDraw context."""
    body_w, body_h = size // 5, size * 3 // 10
    body_top = cy - body_h
    body_bot = cy + body_h // 4
    draw.rounded_rectangle(
        [cx - body_w, body_top, cx + body_w, body_bot],
        radius=body_w, fill=mic_color,
    )
    arc_w = body_w + size // 8
    arc_top = body_top + size // 10
    arc_bot = body_bot + size // 8
    lw = max(2, size // 16)
    draw.arc(
        [cx - arc_w, arc_top, cx + arc_w, arc_bot],
        start=0, end=180, fill=mic_color, width=lw,
    )
    stand_top = arc_bot
    stand_bot = stand_top + size // 7
    draw.line([cx, stand_top, cx, stand_bot], fill=mic_color, width=lw)
    base_w = size // 6
    draw.line([cx - base_w, stand_bot, cx + base_w, stand_bot], fill=mic_color, width=lw)


def create_mic_icon(size: int = 64) -> Image.Image:
    """Green-circle mic icon — idle state."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = 2
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(46, 204, 113))
    _draw_mic(draw, size // 2, size // 2, size)
    return img


def create_recording_icon(size: int = 64) -> Image.Image:
    """Red-circle mic icon — recording state."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = 2
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(231, 76, 60))
    _draw_mic(draw, size // 2, size // 2, size)
    return img


def create_transcribing_icon(size: int = 64) -> Image.Image:
    """Amber-circle mic icon — transcribing state."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = 2
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(243, 156, 18))
    _draw_mic(draw, size // 2, size // 2, size)
    return img


# ─── Custom dictionary ──────────────────────────────────────────────────────────


def load_dictionary() -> None:
    """Load custom terms from ~/.pushtotalk/dictionary.txt (one term per line)."""
    global custom_dictionary
    if not DICTIONARY_FILE.exists():
        log("📖", f"No custom dictionary at {DICTIONARY_FILE} (create one to improve accuracy)")
        return
    try:
        lines = DICTIONARY_FILE.read_text(encoding="utf-8").strip().splitlines()
        custom_dictionary = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
        log("📖", f"Loaded {len(custom_dictionary)} custom dictionary terms")
    except OSError as exc:
        log("❌", f"Failed to load dictionary: {exc}")


# ─── Whisper model ───────────────────────────────────────────────────────────────


def load_model() -> None:
    """Load the faster-whisper model (used as fallback when Groq API is unavailable)."""
    global model
    if GROQ_API_KEY:
        log("☁️", "Groq API key found — using cloud transcription (whisper-large-v3-turbo)")
        log("ℹ️", "Local model will load as fallback …")
    log("⏳", f"Loading Whisper model '{WHISPER_MODEL}' (faster-whisper, int8) …")
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    log("✅", "Model loaded and ready.")


# ─── Cloud transcription (Groq Whisper API) ─────────────────────────────────────


def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert numpy float32 audio array to WAV file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def _build_multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Build multipart/form-data body. files: {name: (filename, data, content_type)}."""
    boundary = f"----PushToTalk{int(time.time() * 1000)}"
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}".encode())
    for key, (filename, data, content_type) in files.items():
        header = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{key}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        )
        parts.append(header.encode() + data)
    parts.append(f"--{boundary}--".encode())
    body = b"\r\n".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def transcribe_groq(audio: np.ndarray) -> str | None:
    """Transcribe audio using Groq's Whisper API. Returns text or None on failure."""
    if not GROQ_API_KEY:
        return None

    try:
        wav_bytes = _audio_to_wav_bytes(audio)

        fields = {
            "model": "whisper-large-v3-turbo",
            "response_format": "json",
            "language": LANGUAGE or "",
            "temperature": "0",
        }
        # Feed custom dictionary as Whisper prompt to bias recognition
        if custom_dictionary:
            fields["prompt"] = ", ".join(custom_dictionary)

        body, content_type = _build_multipart(
            fields=fields,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": content_type,
            },
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            return result.get("text", "").strip()

    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
        log("⚠️", f"Groq API failed, falling back to local: {exc}")
        return None


# ─── LLM cleanup (Claude Haiku) ─────────────────────────────────────────────────


def llm_cleanup(text: str) -> str:
    """Post-process transcript with Claude Haiku: punctuation, filler removal, formatting."""
    if not ANTHROPIC_API_KEY:
        return text

    try:
        dict_hint = ""
        if custom_dictionary:
            dict_hint = f"\nCustom terms to preserve exactly: {', '.join(custom_dictionary)}\n"

        prompt = (
            "You are a speech-to-text post-processor. Clean up this raw voice transcription.\n"
            "\n"
            "Rules:\n"
            "- Remove filler words: um, uh, like, you know, I mean, basically, actually (when used as filler)\n"
            "- Add proper punctuation and capitalization\n"
            "- Resolve self-corrections: e.g. 'meet tomorrow no wait Friday' → 'meet on Friday'\n"
            "- Interpret spoken formatting: 'new line' → newline, 'new paragraph' → double newline,\n"
            "  'period' → '.', 'comma' → ',', 'question mark' → '?', 'exclamation point' → '!',\n"
            "  'colon' → ':', 'semicolon' → ';', 'open quote'/'close quote' → quotes\n"
            "- Keep the tone natural — don't over-formalize casual speech\n"
            "- Do NOT add any explanation, preamble, or commentary\n"
            "- Output ONLY the cleaned text, nothing else\n"
            f"{dict_hint}\n"
            f"Raw transcript: {text}"
        )

        body = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            content = result.get("content", [])
            if content and content[0].get("type") == "text":
                cleaned = content[0]["text"].strip()
                if cleaned:
                    return cleaned

    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError, KeyError) as exc:
        log("⚠️", f"Claude cleanup failed, using raw transcript: {exc}")

    return text


# ─── Voice commands ──────────────────────────────────────────────────────────────

# Meta-commands: if the ENTIRE transcript matches, execute a special action
META_COMMANDS = {
    "scratch that", "scratch this", "undo that", "undo this",
    "never mind", "nevermind", "cancel that", "cancel this",
}

# Text replacement commands (used only when LLM cleanup is disabled)
TEXT_COMMANDS = [
    (r"\bnew paragraph\b", "\n\n"),
    (r"\bnew line\b", "\n"),
    (r"\bperiod\b", "."),
    (r"\bfull stop\b", "."),
    (r"\bcomma\b", ","),
    (r"\bquestion mark\b", "?"),
    (r"\bexclamation point\b", "!"),
    (r"\bexclamation mark\b", "!"),
    (r"\bcolon\b", ":"),
    (r"\bsemicolon\b", ";"),
    (r"\bopen quote\b", '"'),
    (r"\bclose quote\b", '"'),
    (r"\bhyphen\b", "-"),
    (r"\bellipsis\b", "..."),
]


def process_voice_commands(text: str) -> tuple[str | None, bool]:
    """
    Process voice commands in transcript.
    Returns (processed_text, is_meta_command).
    If is_meta_command is True, processed_text is the command name.
    """
    normalized = text.strip().lower().rstrip(".")

    # Check meta-commands (entire utterance is a command)
    if normalized in META_COMMANDS:
        return normalized, True

    # Text replacement commands — only if LLM cleanup is off
    if not ANTHROPIC_API_KEY:
        result = text
        for pattern, replacement in TEXT_COMMANDS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        # Clean up spacing around punctuation
        result = re.sub(r"\s+([.,!?:;])", r"\1", result)
        return result, False

    return text, False


def execute_meta_command(command: str) -> None:
    """Execute a meta voice command like 'scratch that'."""
    if command in ("scratch that", "scratch this", "undo that", "undo this"):
        log("↩️", "Executing: scratch that (undo last paste)")
        time.sleep(0.1)
        if platform.system() == "Darwin":
            pyautogui.hotkey("command", "z")
        else:
            pyautogui.hotkey("ctrl", "z")

    elif command in ("never mind", "nevermind", "cancel that", "cancel this"):
        log("🚫", "Cancelled — no text typed")


# ─── Recording ───────────────────────────────────────────────────────────────────


def get_input_devices() -> list[dict]:
    """Return a list of available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append({"index": i, "name": d["name"], "channels": d["max_input_channels"]})
    return inputs


def set_input_device(index) -> None:
    """Set the active input device by index (None for system default)."""
    global selected_device
    selected_device = index
    if index is None:
        name = "System Default"
        log("🎙️", "Microphone set to system default")
    else:
        name = sd.query_devices(index)["name"]
        log("🎙️", f"Microphone set to: {name}")
    broadcast_event("device", {"index": index, "name": name})


def _update_tray_icon(icon_fn):
    """Update the system tray icon (best-effort)."""
    if tray_icon is not None:
        try:
            tray_icon.icon = icon_fn()
        except Exception:
            pass


def start_recording() -> None:
    """Begin capturing audio from the selected microphone."""
    global recording, audio_frames, stream, record_start_time
    if recording:
        return

    audio_frames = []
    recording = True
    record_start_time = time.time()
    set_state("recording")
    beep(frequency=1000, duration_ms=60)

    def callback(indata, frames, time_info, status):
        if status:
            log("⚠️", f"Audio status: {status}")
        audio_frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=selected_device,
        callback=callback,
    )
    stream.start()
    log("🔴", "Recording … (release key to stop)")
    _update_tray_icon(create_recording_icon)


def stop_recording_and_transcribe() -> None:
    """Stop recording, transcribe, and type the result."""
    global recording, stream
    if not recording:
        return

    recording = False
    duration_s = time.time() - record_start_time if record_start_time else 0

    if stream is not None:
        stream.stop()
        stream.close()
        stream = None

    beep(frequency=600, duration_ms=60)
    set_state("transcribing")
    log("⏹", "Recording stopped. Transcribing …")
    _update_tray_icon(create_transcribing_icon)

    if not audio_frames:
        log("⚠️", "No audio captured.")
        set_state("idle")
        _update_tray_icon(create_mic_icon)
        return

    frames_copy = list(audio_frames)
    threading.Thread(target=_transcribe_and_type, args=(frames_copy, duration_s), daemon=True).start()


def _transcribe_and_type(frames: list, duration_s: float) -> None:
    """Transcribe audio frames and simulate keyboard typing. Full pipeline:
    1. ASR (Groq cloud → local fallback)
    2. Voice command detection
    3. LLM cleanup (Claude Haiku)
    4. Type into focused field
    """
    global last_typed_text
    try:
        audio = np.concatenate(frames, axis=0).flatten()

        if len(audio) < SAMPLE_RATE * 0.3:
            log("⚠️", "Recording too short, skipping.")
            set_state("idle")
            _update_tray_icon(create_mic_icon)
            return

        audio_len = len(audio) / SAMPLE_RATE
        log("🔄", f"Starting transcription ({audio_len:.1f}s of audio) …")

        # ── Step 1: ASR ──────────────────────────────────────────────────
        t0 = time.time()
        engine_used = "local"

        if GROQ_API_KEY:
            broadcast_event("pipeline", {"step": "transcribing", "detail": "Groq cloud"})
            text = transcribe_groq(audio)
            if text is not None:
                engine_used = "groq"
            else:
                # Fallback to local
                broadcast_event("pipeline", {"step": "transcribing", "detail": "Local (fallback)"})
                text = _transcribe_local(audio)
        else:
            broadcast_event("pipeline", {"step": "transcribing", "detail": f"Local ({WHISPER_MODEL})"})
            text = _transcribe_local(audio)

        asr_time = time.time() - t0
        log("⏱️", f"ASR took {asr_time:.1f}s via {engine_used}")

        if not text:
            log("⚠️", "No speech detected.")
            set_state("idle")
            _update_tray_icon(create_mic_icon)
            return

        raw_text = text
        log("📝", f"Raw transcript: {raw_text}")

        # ── Step 2: Voice commands ───────────────────────────────────────
        text, is_meta = process_voice_commands(text)

        if is_meta:
            log("🗣️", f"Voice command: {text}")
            broadcast_event("pipeline", {"step": "command", "detail": text})

            entry = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "text": f"[Command: {text}]",
                "raw_text": raw_text,
                "duration_s": round(duration_s, 1),
                "engine": engine_used,
            }
            transcription_history.append(entry)
            broadcast_event("transcription", entry)

            set_state("idle")
            _update_tray_icon(create_mic_icon)
            execute_meta_command(text)
            return

        # ── Step 3: LLM cleanup ──────────────────────────────────────────
        if ANTHROPIC_API_KEY:
            broadcast_event("pipeline", {"step": "cleaning", "detail": "Claude Haiku"})
            t1 = time.time()
            text = llm_cleanup(text)
            log("✨", f"LLM cleanup took {time.time() - t1:.1f}s")

        log("✅", f"Final text: {text}")
        log_transcription(text)
        send_notification("Push-to-Talk", text[:80])

        # Add to history and broadcast to web UI
        entry = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "text": text,
            "raw_text": raw_text if raw_text != text else None,
            "duration_s": round(duration_s, 1),
            "engine": engine_used,
        }
        transcription_history.append(entry)
        broadcast_event("transcription", entry)

        broadcast_event("pipeline", {"step": "done"})
        set_state("idle")
        _update_tray_icon(create_mic_icon)

        # Type the text into the focused field
        last_typed_text = text
        _type_text(text)

    except Exception as exc:
        log("❌", f"Transcription error: {exc}")
        set_state("idle")
        _update_tray_icon(create_mic_icon)


def _transcribe_local(audio: np.ndarray) -> str:
    """Transcribe audio using the local faster-whisper model."""
    kwargs = {
        "language": LANGUAGE,
        "beam_size": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    if custom_dictionary:
        kwargs["initial_prompt"] = ", ".join(custom_dictionary)

    segments, info = model.transcribe(audio, **kwargs)
    return " ".join(seg.text for seg in segments).strip()


def _type_text(text: str) -> None:
    """Paste text into the currently focused field via clipboard for instant output."""
    time.sleep(0.15)
    try:
        old_clipboard = None
        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            pass

        pyperclip.copy(text)
        time.sleep(0.05)

        if platform.system() == "Darwin":
            pyautogui.hotkey("command", "v")
        else:
            pyautogui.hotkey("ctrl", "v")

        if old_clipboard is not None:
            time.sleep(0.3)
            try:
                pyperclip.copy(old_clipboard)
            except Exception:
                pass
    except Exception as exc:
        log("❌", f"Typing error: {exc}")


# ─── Global hotkey ───────────────────────────────────────────────────────────────


def setup_hotkey() -> None:
    """Register global hotkey press/release handlers using pynput."""
    global key_listener

    def on_press(key):
        if key == HOTKEY:
            start_recording()

    def on_release(key):
        if key == HOTKEY:
            stop_recording_and_transcribe()

    key_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    key_listener.daemon = True
    key_listener.start()
    log("🎤", f"Hotkey '{HOTKEY}' registered. Hold to record, release to transcribe.")


# ─── System tray ─────────────────────────────────────────────────────────────────


def on_quit(icon, item) -> None:
    """Clean up and exit."""
    log("👋", "Quitting …")
    icon.stop()
    if key_listener is not None:
        key_listener.stop()
    os._exit(0)


def _build_mic_menu() -> pystray.Menu:
    """Build a submenu listing all input devices for microphone selection."""
    devices = get_input_devices()

    def make_setter(idx):
        def on_click(icon, item):
            set_input_device(idx)
            icon.menu = _build_tray_menu()
        return on_click

    items = [
        pystray.MenuItem(
            "System Default",
            make_setter(None),
            checked=lambda item: selected_device is None,
        )
    ]
    for d in devices:
        idx = d["index"]
        items.append(
            pystray.MenuItem(
                d["name"],
                make_setter(idx),
                checked=lambda item, _idx=idx: selected_device == _idx,
            )
        )
    return pystray.Menu(*items)


def _build_tray_menu() -> pystray.Menu:
    """Build the full tray menu."""
    engine_label = "Groq Cloud" if GROQ_API_KEY else f"Local ({WHISPER_MODEL})"
    return pystray.Menu(
        pystray.MenuItem(APP_NAME, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Hotkey: Option/Alt (hold to talk)", None, enabled=False),
        pystray.MenuItem(f"Engine: {engine_label}", None, enabled=False),
        pystray.MenuItem(f"Cleanup: {'Claude Haiku' if ANTHROPIC_API_KEY else 'Off'}", None, enabled=False),
        pystray.MenuItem(f"Web UI: http://localhost:{WEB_PORT}", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Microphone", _build_mic_menu()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )


def setup_tray() -> None:
    """Create and run the system-tray icon (blocks the calling thread)."""
    global tray_icon
    tray_icon = pystray.Icon(
        APP_NAME,
        icon=create_mic_icon(),
        title=APP_NAME,
        menu=_build_tray_menu(),
    )
    tray_icon.run()


# ─── Web server ──────────────────────────────────────────────────────────────────


def start_web_server() -> None:
    """Start the Flask web UI on a background daemon thread."""
    thread = threading.Thread(
        target=lambda: flask_app.run(host="127.0.0.1", port=WEB_PORT, threaded=True),
        daemon=True,
    )
    thread.start()
    log("🌐", f"Web UI available at http://localhost:{WEB_PORT}")


# ─── Entry point ─────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 50)
    print(f"  {APP_NAME} — Voice to Text")
    print("=" * 50)
    print()

    # Set macOS Dock icon and app name
    if platform.system() == "Darwin":
        try:
            from AppKit import NSApplication, NSImage
            from Foundation import NSBundle

            bundle = NSBundle.mainBundle()
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info:
                info["CFBundleName"] = APP_NAME

            app = NSApplication.sharedApplication()
            icon_img = create_mic_icon(size=256)
            buf = io.BytesIO()
            icon_img.save(buf, format="PNG")
            ns_image = NSImage.alloc().initWithData_(buf.getvalue())
            app.setApplicationIconImage_(ns_image)
        except ImportError:
            pass

    # Load custom dictionary
    load_dictionary()

    # Show configuration
    if GROQ_API_KEY:
        log("☁️", "Transcription: Groq (whisper-large-v3-turbo)")
    else:
        log("💻", f"Transcription: Local ({WHISPER_MODEL})")

    if ANTHROPIC_API_KEY:
        log("✨", "LLM cleanup: Claude Haiku (enabled)")
    else:
        log("ℹ️", "LLM cleanup: Off (set ANTHROPIC_API_KEY to enable)")

    # Load Whisper model
    load_model()

    # Register global hotkey
    setup_hotkey()

    # Start web UI server
    start_web_server()

    print()
    log("🟢", "Ready! Hold the hotkey to record, release to transcribe.")
    log("ℹ️", "The app is running in the system tray. Right-click the icon to quit.")
    log("🌐", f"Open http://localhost:{WEB_PORT} for the web UI.")

    if platform.system() == "Darwin":
        print()
        print("  ⚠️  macOS Accessibility Permission Required:")
        print("     System Settings → Privacy & Security → Accessibility")
        print("     Add and enable your terminal app (Terminal, iTerm2, etc.)")
        print("     Then restart this app.")
    print()

    # Run tray icon on the main thread (required by some backends)
    setup_tray()


if __name__ == "__main__":
    main()
