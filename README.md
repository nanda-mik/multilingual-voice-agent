# Multilingual Voice Agent Workshop

Build a voice agent that listens, thinks, and talks back -- in 10+ Indian languages.

```
Microphone → VAD → STT → LLM → TTS → Speaker
```

| Component | Tech                   | What it does                       |
|-----------|------------------------|------------------------------------|
| VAD       | Silero VAD (local)     | Trims silence, keeps only speech   |
| STT       | Sarvam AI saarika:v2.5 | Transcribes speech to text         |
| LLM       | Google Gemini 2.0 Flash| Generates a conversational reply   |
| TTS       | Sarvam AI bulbul:v3    | Synthesizes the reply as speech    |
| UI        | Gradio                 | Mic input + audio playback         |

---

## Quick Start

### 1. Clone and enter the directory

```bash
cd voice-agent-workshop
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is a large download (~2 GB). If you already have PyTorch installed, you can skip it in the requirements.

### 4. Set up API keys

```bash
cp .env.example .env
```

Edit `.env` and paste your real keys:

| Key              | Where to get it                          |
|------------------|------------------------------------------|
| `SARVAM_API_KEY` | https://dashboard.sarvam.ai              |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey       |

### 5. Launch

```bash
python app.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

---

## How It Works

### File structure

```
voice-agent-workshop/
├── requirements.txt            # Dependencies
├── .env.example                # API key template
├── pipeline.py                 # Core: VAD + STT + LLM + TTS + orchestrator
├── app.py                      # Gradio UI (entry point)
├── homework_tool_calling.py    # Homework: add tool calling to the agent
└── README.md                   # You are here
```

### pipeline.py (the core)

| Function / Class          | Lines  | Purpose                                |
|---------------------------|--------|----------------------------------------|
| `VoiceActivityDetector`   | ~40    | Load Silero VAD, trim silence from audio |
| `transcribe()`            | ~15    | Call Sarvam STT API                     |
| `ask_llm()`               | ~20    | Call Gemini Flash with conversation history |
| `synthesize()`            | ~25    | Call Sarvam TTS streaming API           |
| `run_pipeline()`          | ~30    | Chain all four, log timing per step     |

### app.py (the UI)

A Gradio Blocks app with a microphone, language dropdown, chatbot display, and auto-playing audio output. Under 100 lines.

---

## Session Agenda (75 min teaching + 15 min Q&A)

| Time  | Min | Topic                                                  |
|-------|-----|--------------------------------------------------------|
| 0:00  | 10  | Intro + Architecture walkthrough                       |
| 0:10  | 10  | Environment setup (clone, pip install, .env)           |
| 0:20  | 8   | Component 1: VAD -- Silero VAD, live-code + demo       |
| 0:28  | 8   | Component 2: STT -- Sarvam saarika, live-code + demo   |
| 0:36  | 8   | Component 3: LLM -- Gemini Flash, live-code + demo     |
| 0:44  | 8   | Component 4: TTS -- Sarvam bulbul streaming, live-code |
| 0:52  | 8   | Pipeline orchestration + latency discussion            |
| 1:00  | 10  | Gradio UI + end-to-end demo                            |
| 1:10  | 5   | Homework brief: tool calling                           |
| 1:15  | 15  | Q&A                                                    |

---

## Homework: Tool Calling

After the workshop, extend the agent so it can **do things**, not just talk.

Run the standalone demo:

```bash
python homework_tool_calling.py
```

Then integrate `ask_llm_with_tools()` into `pipeline.py` by replacing the `ask_llm()` call inside `run_pipeline()`.

See `homework_tool_calling.py` for full instructions and working code.

---

## Supported Languages

Hindi, English, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Odia, Punjabi.

---

## Troubleshooting

| Problem                          | Fix                                                  |
|----------------------------------|------------------------------------------------------|
| `ModuleNotFoundError: torch`     | Run `pip install torch torchaudio`                   |
| VAD model download fails         | Check internet connection; torch hub needs GitHub access |
| "No speech detected"             | Speak louder / closer to mic; check mic permissions  |
| STT returns empty                | Verify `SARVAM_API_KEY` in `.env`                    |
| Gemini 403 error                 | Verify `GEMINI_API_KEY` in `.env`; check quota       |
| TTS returns no audio             | Verify `SARVAM_API_KEY`; check language code match   |
