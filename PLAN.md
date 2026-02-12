# Push-to-Talk Improvement Plan

## What Wispr Flow & Aqua Voice Do Differently

Both apps follow the same core insight: **modern dictation is NOT just transcription**. They run a pipeline:

```
Audio → Fast ASR (Whisper-like) → LLM Post-Processing → Formatted Text
```

The LLM step is what makes them magic. It:
- Removes filler words ("um", "uh", "like")
- Adds punctuation and capitalization from speech cadence
- Resolves mid-sentence corrections ("meet tomorrow, no wait, Friday" → "meet on Friday")
- Formats contextually (bullet lists, paragraphs)
- Strips false positives / hallucinated text

## Proposed Changes (Priority Order)

### 1. Cloud Whisper API via Groq (speed + accuracy)
Replace local `faster-whisper` with Groq's Whisper API (`whisper-large-v3-turbo`).
- Groq returns results in ~0.3-0.5s for typical clips (vs seconds locally)
- `large-v3-turbo` is far more accurate than local `base.en`
- Falls back to local `faster-whisper` if no API key or offline
- Requires: `GROQ_API_KEY` env var

### 2. LLM cleanup pass via Claude API (the big one)
After transcription, send the raw text through Claude Haiku for cleanup:
- Remove filler words and verbal tics
- Add proper punctuation and capitalization
- Resolve self-corrections ("I mean", "actually", "no wait")
- Keep it natural — don't over-formalize
- ~200-400ms added latency (Haiku is fast)
- Requires: `ANTHROPIC_API_KEY` env var
- Skip this step if no API key (just use raw transcript)

### 3. Voice commands
Detect and execute common spoken commands before typing:
- "new line" / "new paragraph" → insert line breaks
- "period" / "comma" / "question mark" → insert punctuation
- "scratch that" / "undo that" → delete last transcription
- Processed locally, no API needed

### 4. Streaming transcription display
Show partial text in the web UI as segments arrive, so the user sees progress immediately instead of waiting for the full result.

### 5. Custom dictionary (config file)
A `~/.pushtotalk/dictionary.txt` file with custom terms:
- Technical jargon: "kubectl", "PyTorch", "useState"
- Names: "Anthropic", your name, coworkers
- Fed as `initial_prompt` to Whisper and as context to the LLM

## Architecture After Changes

```
[Hold hotkey] → Record audio
    ↓
[Release hotkey] → Stop recording
    ↓
[Groq Whisper API] → Raw transcript  (~0.3s)
  (fallback: local faster-whisper)
    ↓
[Voice command check] → Handle "new line", "scratch that", etc.
    ↓
[Claude Haiku API] → Clean, punctuated, formatted text  (~0.3s)
  (skip if no API key)
    ↓
[Clipboard paste] → Instant text insertion
```

**Target end-to-end latency: ~0.6-1.0s** (vs current multi-second local inference)

## What We're NOT Doing (for now)
- Screen capture / context awareness (privacy-heavy, complex)
- Command mode for rewriting selected text (scope creep)
- Personal style learning / RL training
- Real-time streaming ASR during recording
