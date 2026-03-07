# AI Meeting Assistant — Stage 1: Foundation Layer

Real-time microphone capture and speech-to-text transcription using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), running fully
offline on your local machine.

## What this stage does

* Opens the system microphone and captures audio in 5-second chunks
* Transcribes each chunk with a local Whisper model (CTranslate2 backend)
* Prints `[TRANSCRIPT] …` lines to the terminal as speech is detected
* Uses Voice Activity Detection (VAD) to skip silence automatically

No UI, no LLM, no database — just **audio in → text out**.

---

## Setup

```bash
# 1. Create & activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install the project in editable mode
pip install -e .
```

## Configuration

All settings live in the `.env` file at the project root:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | Whisper model size (`tiny.en`, `base.en`, `small.en`, `medium.en`) |
| `WHISPER_DEVICE` | `cuda` | `cuda` for GPU, `cpu` for CPU-only |
| `WHISPER_COMPUTE_TYPE` | `float16` | `float16` for GPU, `int8` for CPU |
| `AUDIO_SAMPLE_RATE` | `16000` | Sample rate in Hz (Whisper expects 16 kHz) |
| `AUDIO_CHUNK_SECONDS` | `5` | Seconds of audio per transcription chunk |
| `AUDIO_INPUT_DEVICE` | *(empty = default)* | Integer device index for the microphone |

> **Tip:** If the wrong mic is being used, list available devices with:
> ```bash
> python -c "import sounddevice; print(sounddevice.query_devices())"
> ```
> Then set `AUDIO_INPUT_DEVICE` to the correct index.

> **No GPU?** Set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8`.

## Running

```bash
python main.py
```

Speak into your microphone — transcripts will appear in the terminal.
Press **Ctrl+C** to stop.

---

## Project Structure

```
├── main.py              # Entry point
├── core/
│   ├── config.py        # Pydantic settings (loads .env)
│   ├── audio_capture.py # Mic input → audio queue
│   ├── transcriber.py   # faster-whisper STT engine
│   └── pipeline.py      # Wires capture + transcriber + callbacks
├── processors/          # (Stage 2 — future)
├── storage/             # (Stage 3 — future)
└── ui/                  # (Stage 4 — future)
```
