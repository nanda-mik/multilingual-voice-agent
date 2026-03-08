# ===================================================================
#  HOMEWORK — Add a Real API Tool (Cricket News or Weather)
# ===================================================================
#
#  Build an inline tool (same pattern as the joke tool in pipeline.py)
#  that calls a real API and speaks the result, bypassing the LLM.
#
#  Steps:
#    1. Define trigger keywords (e.g. "weather", "cricket score", ...)
#    2. Write an intent detector: _is_weather_query(transcript) → bool
#    3. Write an API caller that returns a short TTS-friendly string
#    4. Write _get_weather_reply(transcript, history) → (reply, history) | None
#
#  Integration (pipeline.py → run_pipeline, Step 3):
#    Chain your tool check after _get_joke_reply, before the LLM fallback.
#
# ===================================================================
