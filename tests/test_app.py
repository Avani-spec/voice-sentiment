from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_text_mode(tiny_model_dir, monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL", tiny_model_dir)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and at.info  # prompt shown
    at.text_area[0].input("i loved this movie").run()
    assert not at.exception
    assert {m.label for m in at.metric} >= {"Sentiment score", "Polarity", "Confidence"}


def test_app_bad_model_message(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL", "/nonexistent/model")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.error and "Could not load" in at.error[0].value
