"""
app.py — Voice Agent Gradio UI
================================
This is the entry point.  It builds a web UI with:
  - A microphone button to record your voice
  - A chatbot panel showing the conversation
  - An audio player that auto-plays the agent's response

Run with:
    python app.py
"""
import os

import gradio as gr
from pipeline import run_pipeline

# macOS system proxy causes Python HTTP clients to send absolute-form
# request targets, which uvicorn can't handle. Bypass proxy for localhost.
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")


# -----------------------------------------------------------------
#  Callback — wired to the "Send" button
# -----------------------------------------------------------------

def respond(audio_path, history, chat_display):
    """
    Handle one turn of conversation:
      1. Run the full pipeline (VAD → STT → LLM → TTS)
      2. Update the chatbot display
      3. Return the audio response for playback
    """
    if not audio_path:
        gr.Warning("Please record some audio first!")
        return None, history, chat_display or []

    transcript, reply, tts_audio, updated_history = run_pipeline(
        audio_path, history
    )

    chat_display = chat_display or []
    chat_display.append({"role": "user", "content": transcript})
    if reply:
        chat_display.append({"role": "assistant", "content": reply})

    return tts_audio, updated_history, chat_display


# -----------------------------------------------------------------
#  Build the UI
# -----------------------------------------------------------------

with gr.Blocks(
    title="Multilingual Voice Agent",
) as app:

    gr.Markdown(
        "# Multilingual Voice Agent\n"
        "Speak into your microphone → the agent listens,thinks and talks back."
    )

    # --- Conversation history (hidden state, persists across turns) ---
    history_state = gr.State(value=[])

    with gr.Row():
        # Left column: controls
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record your voice",
            )
            send_btn = gr.Button("Send", variant="primary", size="lg")

        # Right column: conversation + response audio
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
            )
            audio_output = gr.Audio(
                label="Agent response",
                autoplay=True,
            )

    # Wire the button
    send_btn.click(
        fn=respond,
        inputs=[audio_input, history_state, chatbot],
        outputs=[audio_output, history_state, chatbot],
    )


# -----------------------------------------------------------------
#  Launch
# -----------------------------------------------------------------

if __name__ == "__main__":
    app.launch(
        theme=gr.themes.Soft()
    )
