#!/usr/bin/env python3
"""
Push-to-Talk — Hold a hotkey, speak, and have your speech typed into any focused text field.

Similar to Wispr Flow. Runs as a system-tray background app with global hotkey support.
"""

import datetime
import io
import os
import platform
import sys
import threading
import time

import numpy as np
import pyautogui
import pyperclip
import pystray
import sounddevice as sd
import whisper
from PIL import Image, ImageDraw
from pynput import keyboard as pynput_keyboard

# ─── Configuration ──────────────────────────────────────────────────────────────

APP_NAME = "Push to Talk"
HOTKEY = pynput_keyboard.Key.alt_l  # macOS Option key; change to Key.alt_r, Key.cmd, etc. as desired
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
key_listener = None
selected_device = None  # None = system default; set to device index to override

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


def _draw_mic(draw, cx, cy, size, mic_color="white"):
    """Draw a microphone shape onto an ImageDraw context."""
    # Mic body
    body_w, body_h = size // 5, size * 3 // 10
    body_top = cy - body_h
    body_bot = cy + body_h // 4
    draw.rounded_rectangle(
        [cx - body_w, body_top, cx + body_w, body_bot],
        radius=body_w,
        fill=mic_color,
    )
    # Arc around mic
    arc_w = body_w + size // 8
    arc_top = body_top + size // 10
    arc_bot = body_bot + size // 8
    lw = max(2, size // 16)
    draw.arc(
        [cx - arc_w, arc_top, cx + arc_w, arc_bot],
        start=0, end=180,
        fill=mic_color,
        width=lw,
    )
    # Stand
    stand_top = arc_bot
    stand_bot = stand_top + size // 7
    draw.line([cx, stand_top, cx, stand_bot], fill=mic_color, width=lw)
    # Base
    base_w = size // 6
    draw.line([cx - base_w, stand_bot, cx + base_w, stand_bot], fill=mic_color, width=lw)


def create_mic_icon(size: int = 64) -> Image.Image:
    """Generate a green-circle mic icon — clearly visible in the menu bar / tray."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Green circle background
    pad = 2
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(46, 204, 113))
    _draw_mic(draw, size // 2, size // 2, size, mic_color="white")
    return img


def create_recording_icon(size: int = 64) -> Image.Image:
    """Generate a red-circle mic icon to indicate active recording."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Red circle background
    pad = 2
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(231, 76, 60))
    _draw_mic(draw, size // 2, size // 2, size, mic_color="white")
    return img


# ─── Whisper model ───────────────────────────────────────────────────────────────


def load_model() -> None:
    """Load the Whisper model (runs once at startup)."""
    global model
    log("⏳", f"Loading Whisper model '{WHISPER_MODEL}' …")
    model = whisper.load_model(WHISPER_MODEL)
    log("✅", "Model loaded and ready.")


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
        log("🎙️", "Microphone set to system default")
    else:
        name = sd.query_devices(index)["name"]
        log("🎙️", f"Microphone set to: {name}")


def start_recording() -> None:
    """Begin capturing audio from the selected microphone."""
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
        device=selected_device,
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

        # Pass numpy array directly to Whisper — avoids needing ffmpeg
        result = model.transcribe(
            audio,
            language=LANGUAGE,
            fp16=False,
        )

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
            # Rebuild menu to update the check marks
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
    return pystray.Menu(
        pystray.MenuItem(APP_NAME, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Hotkey: Option/Alt (hold to talk)", None, enabled=False),
        pystray.MenuItem(f"Model: {WHISPER_MODEL}", None, enabled=False),
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


# ─── Entry point ─────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 50)
    print(f"  {APP_NAME} — Voice to Text")
    print("=" * 50)
    print()

    # Set macOS Dock icon and app name so it's recognizable
    if platform.system() == "Darwin":
        try:
            from AppKit import NSApplication, NSImage
            from Foundation import NSBundle

            # Set bundle name (shows in Activity Monitor / menu bar)
            bundle = NSBundle.mainBundle()
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info:
                info["CFBundleName"] = APP_NAME

            # Set Dock icon to our green mic icon
            app = NSApplication.sharedApplication()
            icon_img = create_mic_icon(size=256)
            # Convert PIL image to NSImage via PNG bytes
            buf = io.BytesIO()
            icon_img.save(buf, format="PNG")
            ns_image = NSImage.alloc().initWithData_(buf.getvalue())
            app.setApplicationIconImage_(ns_image)
        except ImportError:
            pass  # PyObjC not installed; tray title still works


    # Load Whisper model (may take a moment on first run)
    load_model()

    # Register global hotkey
    setup_hotkey()

    print()
    log("🟢", "Ready! Hold the hotkey to record, release to transcribe.")
    log("ℹ️", "The app is running in the system tray. Right-click the icon to quit.")

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
