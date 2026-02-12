# Push-to-Talk — Voice to Text Desktop Tool

Hold a hotkey, speak, and have your speech typed into any focused text field. Runs as a system-tray background app with offline transcription via OpenAI Whisper.

## Quick Start

```bash
pip3 install -r requirements.txt
python3 pushtotalk.py
```

On first run Whisper will download the model weights (~140 MB for `base`). Subsequent starts are fast.

## How It Works

1. The app sits in your **system tray** as a microphone icon.
2. **Hold the Alt/Option key** to record from your microphone.
3. **Release** the key — audio is transcribed and cleaned up.
4. The transcribed text is **typed into whatever text field is focused** (browser, editor, Slack, etc.).

## Transcription Pipeline

```
Audio → ASR (Groq cloud or local Whisper) → Voice Commands → LLM Cleanup (Claude Haiku) → Paste
```

Each step is optional and degrades gracefully — the app works fully offline with no API keys.

### Cloud Transcription (Groq)

Set `GROQ_API_KEY` to use Groq's `whisper-large-v3-turbo` for fast, accurate transcription (~0.3s). Falls back to local `faster-whisper` if the API is unavailable.

```bash
export GROQ_API_KEY="gsk_..."
```

### LLM Cleanup (Claude Haiku)

Set `ANTHROPIC_API_KEY` to post-process transcripts through Claude Haiku:
- Removes filler words (um, uh, like, you know)
- Adds proper punctuation and capitalization
- Resolves self-corrections ("meet tomorrow no wait Friday" → "meet on Friday")
- Interprets formatting commands ("new line", "period", etc.)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Voice Commands

These spoken commands are recognized and executed:

| Command | Action |
|---|---|
| "scratch that" / "undo that" | Undo the last paste (Cmd/Ctrl+Z) |
| "never mind" / "cancel that" | Cancel — no text typed |
| "new line" / "new paragraph" | Insert line break(s) * |
| "period" / "comma" / "question mark" | Insert punctuation * |

\* Text formatting commands are handled by the LLM when enabled, or by regex when offline.

### Custom Dictionary

Create `~/.pushtotalk/dictionary.txt` with one term per line to improve recognition of technical terms, names, and jargon:

```
kubectl
PyTorch
useState
Anthropic
```

Lines starting with `#` are treated as comments.

## Configuration

Edit the constants at the top of `pushtotalk.py`:

| Variable | Default | Description |
|---|---|---|
| `HOTKEY` | `"alt"` | Key to hold for recording (Option on macOS) |
| `WHISPER_MODEL` | `"base.en"` | Local Whisper model: `tiny.en`, `base.en`, `small.en`, `medium.en`, `large` |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `LANGUAGE` | `"en"` | Language code, or `None` for auto-detect |
| `LOG_FILE` | `"transcription_log.txt"` | Path to log file, or `None` to disable |
| `WEB_PORT` | `8528` | Web UI port |

Environment variables:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for cloud transcription |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM cleanup |

## Web UI

Open `http://localhost:8528` for:
- Live status indicator with pipeline progress
- Microphone selection
- Transcription history with raw/cleaned comparison
- Engine and cleanup status display

## Platform Notes

### macOS
- Grant **Accessibility** and **Microphone** permissions to your terminal / Python.
- The hotkey `"alt"` maps to the **Option** key.

### Windows
- Should work out of the box.
- Run as Administrator if global hotkeys aren't detected.

### Linux
- The `pynput` library may require **root** or `sudo` for global hotkeys.
- Install `xdotool` or `xclip` if clipboard paste doesn't work.

## Dependencies

All listed in `requirements.txt`:

```
sounddevice numpy faster-whisper pynput pyautogui pystray Pillow pyperclip flask
```

No additional dependencies needed for cloud APIs — they use Python's built-in `urllib`.

## License

MIT
