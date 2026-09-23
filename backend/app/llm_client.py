"""
Shared Gemini client. Every agent that needs an LLM call goes through
this function rather than configuring its own client — keeps the API key
and model choice in one place instead of duplicated across agent files.
"""

import google.generativeai as genai

from app.config import settings

genai.configure(api_key=settings.gemini_api_key)

# Flash-Lite: fast, generous free tier, good enough for short classification
# and extraction tasks. Verify this exact model string is still current in
# Google AI Studio (ai.google.dev) — model names get retired/renamed over
# time, so if this errors with a "model not found", check there for the
# current free-tier Flash-Lite model id.
_model = genai.GenerativeModel("gemini-flash-lite-latest")


def generate_text(prompt: str) -> str:
    response = _model.generate_content(prompt)
    return response.text.strip()
  
  
if __name__ == "__main__":
    # Sanity check: python -m app.llm_client
    print(generate_text("Say hello in exactly one word."))