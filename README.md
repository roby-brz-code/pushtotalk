# Push-to-Talk — Voice to Text Desktop Tool

Hold a hotkey, speak, and have your speech typed into any focused text field. Runs as a system-tray background app with offline transcription via OpenAI Whisper.

## Quick Start

```bash
pip install -r requirements.txt
python pushtotalk.py
```

On first run Whisper will download the model weights (~140 MB for `base`). Subsequent starts are fast.

## How It Works

1. The app sits in your **system tray** as a microphone icon.
2. **Hold the Alt/Option key** to record from your microphone.
3. **Release** the key — audio is transcribed locally with Whisper.
4. The transcribed text is **typed into whatever text field is focused** (browser, editor, Slack, etc.).
5. Non-ASCII text (unicode, accented characters) is pasted via the clipboard.

## Configuration

Edit the constants at the top of `pushtotalk.py`:

| Variable | Default | Description |
|---|---|---|
| `HOTKEY` | `"alt"` | Key to hold for recording (Option on macOS) |
| `WHISPER_MODEL` | `"base"` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `LANGUAGE` | `"en"` | Language code, or `None` for auto-detect |
| `LOG_FILE` | `"transcription_log.txt"` | Path to log file, or `None` to disable |

## Platform Notes

### macOS
- Grant **Accessibility** and **Microphone** permissions to your terminal / Python.
- The hotkey `"alt"` maps to the **Option** key.

### Windows
- Should work out of the box.
- Run as Administrator if global hotkeys aren't detected.

### Linux
- The `keyboard` library requires **root** or `sudo`.
- Install `xdotool` or `xclip` if clipboard paste doesn't work.

## Features

- Offline transcription (no API keys, no network)
- System-tray icon with recording indicator
- Audio beep on record start/stop
- Desktop notifications on transcription
- Transcription history logged to a text file
- Clipboard fallback for non-ASCII text

## Dependencies

All listed in `requirements.txt`:

```
sounddevice numpy openai-whisper keyboard pyautogui pystray Pillow scipy pyperclip
```

## License

MIT
