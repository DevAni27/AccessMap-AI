"""LLM client behaviour with Gemini replaced by fakes: retries, JSON parsing, failure mapping."""

import pytest
from google.api_core import exceptions as gexc

from app import llm_client


class FakeResponse:
    def __init__(self, text):
        self._text = text

    @property
    def text(self):
        if self._text is None:
            raise ValueError("blocked")   # what the real client does for blocked answers
        return self._text


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(llm_client._throttle, "wait", lambda: None)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)


def script(monkeypatch, *outcomes):
    """Make generate_content return/raise the given outcomes, in order."""
    calls = []
    queue = list(outcomes)

    def fake(prompt, generation_config=None):
        calls.append(generation_config)
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)

    monkeypatch.setattr(llm_client._model, "generate_content", fake)
    return calls


def test_generate_text_strips_whitespace(monkeypatch):
    script(monkeypatch, "  hello \n")
    assert llm_client.generate_text("hi") == "hello"


def test_json_mode_is_requested_and_parsed(monkeypatch):
    calls = script(monkeypatch, '{"a": 1}')
    assert llm_client.generate_json("p") == {"a": 1}
    assert calls[0].response_mime_type == "application/json"


def test_markdown_fences_are_tolerated(monkeypatch):
    script(monkeypatch, '```json\n{"a": 1}\n```')
    assert llm_client.generate_json("p") == {"a": 1}


def test_invalid_json_gets_one_more_attempt(monkeypatch):
    script(monkeypatch, "not json", '{"ok": true}')
    assert llm_client.generate_json("p") == {"ok": True}


def test_invalid_json_twice_raises_llm_error(monkeypatch):
    script(monkeypatch, "nope", "still nope")
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_json("p")


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    calls = script(monkeypatch, gexc.ResourceExhausted("429"), gexc.ServiceUnavailable("503"), "ok")
    assert llm_client.generate_text("p", retries=3) == "ok"
    assert len(calls) == 3


def test_gives_up_after_retries_with_llm_error(monkeypatch):
    calls = script(monkeypatch, *[gexc.ResourceExhausted("429")] * 4)
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_json("p", retries=3)
    assert len(calls) == 4          # 1 try + 3 retries


def test_generate_text_fails_fast_by_default(monkeypatch):
    calls = script(monkeypatch, gexc.ResourceExhausted("429"), gexc.ResourceExhausted("429"))
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_text("p")
    assert len(calls) == 2          # 1 try + 1 retry


def test_non_transient_errors_are_not_retried(monkeypatch):
    calls = script(monkeypatch, gexc.PermissionDenied("bad key"), "never reached")
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_text("p", retries=3)
    assert len(calls) == 1


def test_blocked_answer_maps_to_llm_error(monkeypatch):
    script(monkeypatch, None)
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_text("p")
