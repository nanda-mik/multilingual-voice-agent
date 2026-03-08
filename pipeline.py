"""
pipeline.py — Voice Agent Pipeline
====================================
This file contains the four core components of our voice agent:

  1. VAD  (Voice Activity Detection) — Silero VAD
  2. STT  (Speech-to-Text)          — Sarvam AI  (saaras:v3)
  3. LLM  (Language Model)          — Google Gemini Flash
  4. TTS  (Text-to-Speech)          — Sarvam AI  (bulbul:v3)

And the orchestrator that chains them together: run_pipeline()

Pipeline flow:
  Microphone → VAD → STT → LLM → TTS → Speaker
"""

import os
import time
import tempfile
import logging
from typing import Optional

import torch
import torchaudio
import requests
from dotenv import load_dotenv
from sarvamai import SarvamAI
from google import genai
from google.genai import types


load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ===================================================================
#  COMPONENT 1 — Voice Activity Detection (Silero VAD)
# ===================================================================
#  https://github.com/snakers4/silero-vad
# -----------------------------------------------------------------

class VoiceActivityDetector:
    """Loads Silero VAD once and reuses it across calls."""

    def __init__(self):
        logger.info("Loading Silero VAD model (first time may download ~1 MB) …")

        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

        # Silero ships five helper functions — we only need three
        (
            self.get_speech_timestamps,  # detects speech segments
            _save_audio,
            self.read_audio,             # reads audio into a tensor
            _VADIterator,
            self.collect_chunks,         # extracts speech chunks
        ) = utils

        logger.info("Silero VAD ready")

    def process(self, audio_path: str) -> str:
        """
        Trim silence from an audio file, keeping only speech.

        Args:
            audio_path: path to a .wav file (from Gradio microphone)

        Returns:
            Path to a trimmed .wav file. If no speech is found, returns
            the original path unchanged.
        """
        # Load audio and convert to 16 kHz mono (what Silero expects)
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:                          # stereo → mono
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != 16000:                            # resample
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

        audio_tensor = waveform.squeeze()                   # shape: (samples,)

        # Ask VAD: "where is the speech?"
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor, self.model, sampling_rate=16000
        )

        if not speech_timestamps:
            logger.warning("VAD found no speech — passing original audio through")
            return audio_path

        # Keep only the speech portions
        speech_audio = self.collect_chunks(speech_timestamps, audio_tensor)

        original_dur = len(audio_tensor) / 16000
        trimmed_dur = len(speech_audio) / 16000
        logger.info(f"VAD trimmed {original_dur:.1f}s → {trimmed_dur:.1f}s")

        # Save to a temporary wav file
        trimmed_path = tempfile.mktemp(suffix=".wav")
        torchaudio.save(trimmed_path, speech_audio.unsqueeze(0), 16000)
        return trimmed_path



# ===================================================================
#  COMPONENT 2 — Speech-to-Text (Sarvam AI — saaras:v3)
# ===================================================================
#  https://docs.sarvam.ai/api-reference-docs/getting-started/models/saaras
# ===================================================================

def transcribe(audio_path: str) -> tuple[str, str]:
    """
    Transcribe an audio file to text using Sarvam STT.

    Args:
        audio_path: path to the audio file (.wav or .mp3)

    Returns:
        (transcript, detected_language_code) — language_code from STT for TTS.
    """
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

    with open(audio_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
        )

    transcript = response.transcript
    language_code = response.language_code
    logger.info(f"STT → \"{transcript}\" in {language_code}")
    return transcript, language_code


# ===================================================================
#  COMPONENT 3 — Language Model (Google Gemini 2.0 Flash)
# ===================================================================
#  https://pypi.org/project/google-genai/#:~:text=.show()-,Chats,-Create%20a%20chat
# ===================================================================

SYSTEM_PROMPT = (
    "You are a friendly, helpful multilingual voice assistant. "
    "You respond in the same language the user speaks to you. "
    "Keep your answers concise (2-3 sentences max) since they will "
    "be spoken aloud via text-to-speech. Be warm and conversational."
)

_genai_client = None


def _get_genai_client() -> genai.Client:
    """Lazy-init the Genai client so we don't hit the API at import time."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def ask_llm(
    transcript: str,
    history: list,
) -> tuple[str, list]:
    """
    Send the user's words to Gemini and get a response.

    Args:
        transcript: what the user said (from STT)
        history:    conversation history — list of
                    {"role": "user"|"model", "parts": ["text"]} dicts

    Returns:
        (reply_text, updated_history)
    """
    client = _get_genai_client()

    genai_history = [
        types.Content(
            role=msg["role"],
            parts=[types.Part(text=p) for p in msg["parts"]],
        )
        for msg in history
    ]

    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
        history=genai_history,
    )

    response = chat.send_message(transcript)
    reply = response.text
    logger.info(f"LLM → \"{reply}\"")

    updated_history = list(history)
    updated_history.append({"role": "user", "parts": [transcript]})
    updated_history.append({"role": "model", "parts": [reply]})

    return reply, updated_history


# ===================================================================
#  COMPONENT 4 — Text-to-Speech (Sarvam AI — bulbul:v3, streaming)
# ===================================================================
#  https://docs.sarvam.ai/api-reference-docs/getting-started/models/bulbul
# ===================================================================

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech/stream"


def synthesize(text: str, language_code: str = "hi-IN") -> str:
    """
    Convert text to speech using Sarvam TTS (streaming).

    Args:
        text:          the text to speak
        language_code: target language — "hi-IN", "en-IN", etc.

    Returns:
        Path to the generated .mp3 file.
    """
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    # You can add pace as a parameter and toggle it here; output will have different speed.
    payload = {
        "text": text.strip(),
        "target_language_code": language_code,
        "speaker": "shubh",
        "model": "bulbul:v3",
        "pace": 1.2,
        "speech_sample_rate": 22050,
        "output_audio_codec": "mp3",
        "enable_preprocessing": True,
    }

    output_path = tempfile.mktemp(suffix=".mp3")
    total_bytes = 0

    with requests.post(
        SARVAM_TTS_URL, headers=headers, json=payload, stream=True
    ) as resp:
        if not resp.ok:
            logger.error(f"TTS API error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)

    logger.info(f"TTS → {total_bytes / 1024:.1f} KB audio saved")
    return output_path


# ===================================================================
#  INLINE TOOL — Joke
# ===================================================================
#  When the user asks for a joke, we return a hardcoded reply and skip
#  the LLM. Same pattern can be used for weather, cricket news, etc.
#  Take-home: replace this with a real API (e.g. cricket news, weather).
# ===================================================================

JOKE_KEYWORDS = (
    "joke", "jokes"
)

# One short joke for the demo (kept brief for TTS).
DEMO_JOKE = (
    "Why did the programmer quit? Because they didn't get arrays."
)


def _is_joke_query(transcript: str) -> bool:
    """True if the user is asking for a joke (simple keyword check)."""
    text = transcript.strip().lower()
    return any(kw in text for kw in JOKE_KEYWORDS)


def _get_joke_reply(transcript: str, history: list) -> Optional[tuple[str, list]]:
    """
    If the user asked for a joke, return (reply, updated_history). Else None.
    When we skip the LLM, we still update history so the next turn has context.
    """
    if not _is_joke_query(transcript):
        return None
    reply = DEMO_JOKE
    logger.info(f"Tool (joke) → \"{reply}\"")
    updated_history = list(history)
    updated_history.append({"role": "user", "parts": [transcript]})
    updated_history.append({"role": "model", "parts": [reply]})
    return reply, updated_history


# ===================================================================
#  PIPELINE ORCHESTRATOR
# ===================================================================
#  Chains all four components:  VAD → STT → LLM → TTS
#  Logs wall-clock time for each step so you can see the latency
#  breakdown in the terminal while demoing.
# ===================================================================

_vad: Optional[VoiceActivityDetector] = None


def _get_vad() -> VoiceActivityDetector:
    """Lazy-load VAD so the model isn't downloaded at import time."""
    global _vad
    if _vad is None:
        _vad = VoiceActivityDetector()
    return _vad


def run_pipeline(
    audio_path: str,
    history: list,
) -> tuple[str, str, Optional[str], list]:
    """
    Run the full voice-agent pipeline end to end.

    Args:
        audio_path: file path from Gradio's microphone widget
        history:    conversation history (Gemini format)

    Returns:
        (user_transcript, llm_reply, tts_audio_path, updated_history)
    """
    if not audio_path:
        return "No audio recorded.", "", None, history

    pipeline_start = time.time()

    # ---- Step 1: VAD ----
    t0 = time.time()
    vad = _get_vad()
    trimmed_path = vad.process(audio_path)
    logger.info(f"  ⏱  VAD  : {time.time() - t0:.2f}s")

    # ---- Step 2: STT ----
    t0 = time.time()
    transcript, language_code = transcribe(trimmed_path)
    language_code = language_code or "en-IN"
    logger.info(f"  ⏱  STT  : {time.time() - t0:.2f}s")

    if not transcript or not transcript.strip():
        return "(could not transcribe)", "", None, history

    # ---- Step 3: Tool check (e.g. joke) or LLM ----
    t0 = time.time()
    tool_result = _get_joke_reply(transcript, history)
    if tool_result is not None:
        reply, updated_history = tool_result
        logger.info(f"  ⏱  LLM  : skipped (tool: joke)")
    else:
        reply, updated_history = ask_llm(transcript, history)
        logger.info(f"  ⏱  LLM  : {time.time() - t0:.2f}s")

    # ---- Step 4: TTS ----
    t0 = time.time()
    tts_audio_path = synthesize(reply, language_code=language_code)
    logger.info(f"  ⏱  TTS  : {time.time() - t0:.2f}s")

    total = time.time() - pipeline_start
    logger.info(f"  ⏱  TOTAL: {total:.2f}s  ✓")

    return transcript, reply, tts_audio_path, updated_history
