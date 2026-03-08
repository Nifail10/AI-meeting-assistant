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

---

## Installation

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 2. Install base dependencies
```bash
pip install -r requirements.txt
```

This installs all dependencies including `llama-cpp-python` with CPU
inference only. If you only need speech transcription and do not require
the LLM processors, this is sufficient.

### 3. Enable GPU acceleration for LLM inference (recommended)

The standard `llama-cpp-python` PyPI wheel does not include CUDA support.
To enable GPU inference on an NVIDIA GPU (required for real-time key point
detection to run without lag), reinstall with the CUDA 12.1 wheel:
```bash
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
    --force-reinstall \
    --no-cache-dir
```

> **Note:** faster-whisper manages its own CUDA runtime via CTranslate2
> and does not require this step.

### 4. Download the Mistral model

Download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (~4.1 GB) from:
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

Place the file in the `models/` directory:
```
models/
└── mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 5. Run the assistant
```bash
python main.py
```

Speak into your microphone. Key points appear in real time.
Press **Ctrl+C** to stop — a summary and clarifying questions will
be generated from the full session transcript.

### Dependency quick-reference

| Package | Purpose | Install note |
|---|---|---|
| `sounddevice` | Microphone capture | Standard PyPI |
| `numpy` | Audio buffer math | Standard PyPI |
| `faster-whisper` | Speech-to-text | Standard PyPI |
| `pydantic` | Data validation | Standard PyPI |
| `pydantic-settings` | `.env` config loading | Standard PyPI |
| `python-dotenv` | `.env` file parsing | Standard PyPI |
| `llama-cpp-python` | Local LLM inference | Requires CUDA wheel for GPU |
