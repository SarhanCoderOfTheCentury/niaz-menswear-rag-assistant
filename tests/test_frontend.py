"""Streamlit integration checks for startup and empty/API-unconfigured flows."""
import pytest
from streamlit.testing.v1 import AppTest

from config import ROOT


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return AppTest.from_file(str(ROOT / "streamlet.py"), default_timeout=30).run()


def test_app_starts_with_document_counts_and_no_exceptions(app):
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric} == {
        "Knowledge documents": "5", "Searchable passages": "48", "Policy versions": "2"}
    assert any("GEMINI_API_KEY" in message.value for message in app.info)


def test_empty_input_has_helpful_message(app):
    next(button for button in app.button if button.label == "Ask Niaz").click().run()
    assert not app.exception
    assert any("Type a question" in message.value for message in app.warning)


def test_question_without_api_key_shows_configuration_help(app):
    next(field for field in app.text_input if field.label == "Ask your knowledge base").set_value("What is the return policy?")
    next(button for button in app.button if button.label == "Ask Niaz").click().run()
    assert not app.exception
    assert any("Gemini API key" in message.value for message in app.info)


def test_api_failure_shows_safe_helpful_message(monkeypatch):
    import providers
    from providers import ProviderError
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-not-a-real-key")
    class FailingProvider:
        def __init__(self, *args):
            pass
        def embed(self, *args):
            raise ProviderError("Gemini quota or rate limit reached. Wait and retry, or check your AI Studio quota.")
        def generate(self, *args):
            raise ProviderError("Gemini quota or rate limit reached. Wait and retry, or check your AI Studio quota.")
    monkeypatch.setattr(providers, "GeminiProvider", FailingProvider)
    app = AppTest.from_file(str(ROOT / "streamlet.py"), default_timeout=30).run()
    next(field for field in app.text_input if field.label == "Ask your knowledge base").set_value("What is the price?")
    next(button for button in app.button if button.label == "Ask Niaz").click().run()
    assert not app.exception
    assert any("quota" in message.value for message in app.error)
    assert not app.session_state["messages"]
