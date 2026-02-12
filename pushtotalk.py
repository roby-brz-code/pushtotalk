#!/usr/bin/env python3
"""
Push-to-Talk — Hold a hotkey, speak, and have your speech typed into any focused text field.

Similar to Wispr Flow. Runs as a system-tray background app with global hotkey support.
"""

import datetime
import io
import os
import platform
import struct
import sys
import tempfile
import threading
import time
import wave

import keyboard
import numpy as np
import pyautogui
import pyperclip
import pystray
import sounddevice as sd
import whisper
from PIL import Image, ImageDraw
from scipy.io.wavfile import write as wav_write

# ─── Configuration ──────────────────────────────────────────────────────────────

HOTKEY = "alt"              # macOS Option key = "alt"; use "alt" on Windows/Linux too
WHISPER_MODEL = "base"      # tiny | base | small | medium | large
SAMPLE_RATE = 16000         # 16 kHz mono
LANGUAGE = "en"             # Set to None for auto-detect
LOG_FILE = "transcription_log.txt"  # Set to None to disable logging

# ─── Globals ────────────────────────────────────────────────────────────────────

model = None
recording = False
audio_frames = []
stream = None
tray_icon = None

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


def beep(frequency: int = 800, duration_ms: int = 80) -> None:
    """Play a short sine-wave beep (non-blocking)."""
    try:
        t = np.linspace(0, duration_ms / 1000, int(SAMPLE_RATE * duration_ms / 1000), dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * frequency * t)
        sd.play(tone, samplerate=SAMPLE_RATE)
    except Exception:
        pass  # audio feedback is best-effort


def send_notification(title: str, message: str) -> None:
    """Send a desktop notification (best-effort, cross-platform)."""
    system = platform.system()
    try:
        if system == "Darwin":
            os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
        elif system == "Linux":
            os.system(f'notify-send "{title}" "{message}"')
        elif system == "Windows":
            # Use a quick PowerShell toast; requires Windows 10+
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
        pass  # notifications are best-effort


# ─── Microphone icon generation ─────────────────────────────────────────────────


def create_mic_icon(color: str = "white", bg: str = "black", size: int = 64) -> Image.Image:
    """Generate a simple microphone icon with Pillow."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    # Mic body (rounded rect approximated by ellipse + rect)
    body_w, body_h = size // 4, size // 3
    draw.rounded_rectangle(
        [cx - body_w, cy - body_h, cx + body_w, cy + body_h // 3],
        radius=body_w,
        fill=color,
    )
    # Arc around mic
    arc_w = body_w + size // 10
    arc_top = cy - body_h + size // 10
    arc_bot = cy + body_h // 3 + size // 8
    draw.arc(
        [cx - arc_w, arc_top, cx + arc_w, arc_bot],
        start=0, end=180,
        fill=color,
        width=max(2, size // 20),
    )
    # Stand
    stand_top = arc_bot
    stand_bot = stand_top + size // 6
    draw.line([cx, stand_top, cx, stand_bot], fill=color, width=max(2, size // 20))
    # Base
    base_w = size // 6
    draw.line([cx - base_w, stand_bot, cx + base_w, stand_bot], fill=color, width=max(2, size // 20))

    return img


def create_recording_icon(size: int = 64) -> Image.Image:
    """Generate a red-circle recording icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = size // 6
    draw.ellipse([margin, margin, size - margin, size - margin], fill="red")
    return img


# ─── Whisper model ───────────────────────────────────────────────────────────────


def load_model() -> None:
    """Load the Whisper model (runs once at startup)."""
    global model
    log("⏳", f"Loading Whisper model '{WHISPER_MODEL}' …")
    model = whisper.load_model(WHISPER_MODEL)
    log("✅", "Model loaded and ready.")


# ─── Recording ───────────────────────────────────────────────────────────────────


def start_recording() -> None:
    """Begin capturing audio from the default microphone."""
    global recording, audio_frames, stream
    if recording:
        return

    audio_frames = []
    recording = True
    beep(frequency=1000, duration_ms=60)

    def callback(indata, frames, time_info, status):
        if status:
            log("⚠️", f"Audio status: {status}")
        audio_frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    )
    stream.start()
    log("🔴", "Recording … (release key to stop)")

    # Update tray icon to recording indicator
    if tray_icon is not None:
        try:
            tray_icon.icon = create_recording_icon()
        except Exception:
            pass


def stop_recording_and_transcribe() -> None:
    """Stop recording, transcribe, and type the result."""
    global recording, stream
    if not recording:
        return

    recording = False
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None

    beep(frequency=600, duration_ms=60)
    log("⏹", "Recording stopped. Transcribing …")

    # Restore tray icon
    if tray_icon is not None:
        try:
            tray_icon.icon = create_mic_icon()
        except Exception:
            pass

    if not audio_frames:
        log("⚠️", "No audio captured.")
        return

    # Concatenate and transcribe in a background thread so we don't block the hotkey listener
    frames_copy = list(audio_frames)
    threading.Thread(target=_transcribe_and_type, args=(frames_copy,), daemon=True).start()


def _transcribe_and_type(frames: list) -> None:
    """Transcribe audio frames and simulate keyboard typing."""
    try:
        audio = np.concatenate(frames, axis=0).flatten()

        # Skip very short recordings (< 0.3s)
        if len(audio) < SAMPLE_RATE * 0.3:
            log("⚠️", "Recording too short, skipping.")
            return

        # Write to a temporary WAV file for Whisper
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            wav_write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
            result = model.transcribe(
                tmp.name,
                language=LANGUAGE,
                fp16=False,
            )
        finally:
            tmp.close()
            os.unlink(tmp.name)

        text = result.get("text", "").strip()
        if not text:
            log("⚠️", "No speech detected.")
            return

        log("✅", f"Transcribed: {text}")
        log_transcription(text)
        send_notification("Push-to-Talk", text[:80])

        # Type the text into the focused field
        _type_text(text)

    except Exception as exc:
        log("❌", f"Transcription error: {exc}")


def _type_text(text: str) -> None:
    """Type text into the currently focused field. Falls back to clipboard paste for non-ASCII."""
    # Small delay to let the user release the hotkey before we start typing
    time.sleep(0.15)

    try:
        # Check if text is plain ASCII — pyautogui handles ASCII reliably
        if text.isascii():
            pyautogui.typewrite(text, interval=0.01)
        else:
            # Non-ASCII: use clipboard paste
            old_clipboard = None
            try:
                old_clipboard = pyperclip.paste()
            except Exception:
                pass

            pyperclip.copy(text)
            time.sleep(0.05)

            system = platform.system()
            if system == "Darwin":
                pyautogui.hotkey("command", "v")
            else:
                pyautogui.hotkey("ctrl", "v")

            # Restore previous clipboard after a short delay
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
    """Register global hotkey press/release handlers."""
    keyboard.on_press_key(HOTKEY, lambda e: start_recording(), suppress=False)
    keyboard.on_release_key(HOTKEY, lambda e: stop_recording_and_transcribe(), suppress=False)
    log("🎤", f"Hotkey '{HOTKEY}' registered. Hold to record, release to transcribe.")


# ─── System tray ─────────────────────────────────────────────────────────────────


def on_quit(icon, item) -> None:
    """Clean up and exit."""
    log("👋", "Quitting …")
    icon.stop()
    keyboard.unhook_all()
    os._exit(0)


def setup_tray() -> None:
    """Create and run the system-tray icon (blocks the calling thread)."""
    global tray_icon

    menu = pystray.Menu(
        pystray.MenuItem(f"Hotkey: {HOTKEY} (hold to talk)", None, enabled=False),
        pystray.MenuItem(f"Model: {WHISPER_MODEL}", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    tray_icon = pystray.Icon(
        "PushToTalk",
        icon=create_mic_icon(),
        title="Push-to-Talk",
        menu=menu,
    )

    tray_icon.run()


# ─── Entry point ─────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 50)
    print("  Push-to-Talk — Voice to Text")
    print("=" * 50)
    print()

    # Load Whisper model (may take a moment on first run)
    load_model()

    # Register global hotkey
    setup_hotkey()

    print()
    log("🟢", "Ready! Hold the hotkey to record, release to transcribe.")
    log("ℹ️", "The app is running in the system tray. Right-click the icon to quit.")
    print()

    # Run tray icon on the main thread (required by some backends)
    setup_tray()


if __name__ == "__main__":
    main()
