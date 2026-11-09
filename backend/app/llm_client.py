"""
Gemini client — the only place in the project that talks to the LLM.

Free-tier reality this module is built around:
  * ~15 requests/minute  -> every call goes through one shared throttle.
  * Occasional 429 / 503 -> transient errors are retried with backoff.
  * Output can be blocked, empty or malformed -> every failure surfaces as
    a single exception type, LLMError, so agents can degrade gracefully
    instead of crashing a whole assessment.
"""

import json
import re
import time

import google.generativeai as genai
from google.api_core import exceptions as gexc

from app.config import settings
from app.http_utils import Throttle

MODEL_NAME = "gemini-flash-lite-latest"

genai.configure(api_key=settings.gemini_api_key)
_model = genai.GenerativeModel(MODEL_NAME)

# JSON mode makes Gemini return syntactically valid JSON; temperature 0 makes
# extraction repeatable (same input -> same output).
_JSON_CONFIG = genai.GenerationConfig(response_mime_type="application/json", temperature=0)

_throttle = Throttle(4.0)  # 4 s between calls ~= 15 requests/minute

_RETRYABLE = (
    gexc.ResourceExhausted,    # 429: rate limit
    gexc.ServiceUnavailable,   # 503
    gexc.DeadlineExceeded,     # 504 / timeout
    gexc.InternalServerError,  # 500
)
_BACKOFF_SECONDS = (5, 20, 45)


class LLMError(RuntimeError):
    """The LLM could not produce a usable answer (after retries)."""


def _generate(prompt: str, generation_config, retries: int) -> str:
    backoffs = _BACKOFF_SECONDS[:retries]
    last_error: Exception | None = None

    for attempt in range(len(backoffs) + 1):
        _throttle.wait()
        try:
            response = _model.generate_content(prompt, generation_config=generation_config)
            return response.text.strip()
        except _RETRYABLE as e:
            last_error = e
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])
        except ValueError as e:
            # response.text raises ValueError when the answer was blocked or empty
            raise LLMError(f"Gemini returned no usable text: {e}") from e
        except gexc.GoogleAPICallError as e:
            # not transient (bad key, bad request, permission...): retrying won't help
            raise LLMError(f"Gemini request failed: {e}") from e

    raise LLMError(f"Gemini unavailable after {len(backoffs) + 1} attempts: {last_error}")


def generate_text(prompt: str, *, retries: int = 1) -> str:
    """Free-text answer. Few retries by default: callers (e.g. category
    classification) sit inside user-facing requests and should fail fast."""
    return _generate(prompt, None, retries)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def generate_json(prompt: str, *, retries: int = 3):
    """Answer parsed as JSON (a dict or list). Asks once more if the first
    answer is not valid JSON; raises LLMError if it still is not."""
    last_error: Exception | None = None
    for _ in range(2):
        raw = _generate(prompt, _JSON_CONFIG, retries)
        try:
            return json.loads(_FENCE.sub("", raw.strip()))
        except json.JSONDecodeError as e:
            last_error = e
    raise LLMError(f"Gemini did not return valid JSON: {last_error}")


if __name__ == "__main__":
    # Sanity check: python -m app.llm_client
    print(generate_text("Say hello in exactly one word."))
    print(generate_json('Return the JSON object {"ok": true}.'))
