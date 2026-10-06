"""Streamlit app: record or upload speech, get sentiment.  streamlit run app.py"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from voicesent.audio import duration, load_audio  # noqa: E402
from voicesent.predict import SentimentModel, VoiceSentiment  # noqa: E402

st.set_page_config(page_title="Voice sentiment", layout="wide")
st.title("Voice sentiment")

DEFAULT_MODEL = os.environ.get("SENTIMENT_MODEL", "runs/roberta-mosi/best")
with st.sidebar:
    model_id = st.text_input("Sentiment model (folder or Hugging Face id)", DEFAULT_MODEL,
                             help="Your fine-tuned model from voicesent.train, or a 3-class "
                                  "classifier such as cardiffnlp/twitter-roberta-base-sentiment-latest.")
    asr_id = st.selectbox("Whisper model", ["openai/whisper-base", "openai/whisper-small", "openai/whisper-medium"], index=1)
    mc = st.slider("Uncertainty samples", 1, 50, 20,
                   help="Number of Monte-Carlo dropout passes used to estimate confidence.")


@st.cache_resource(show_spinner="Loading sentiment model...")
def get_sentiment(m):
    return SentimentModel(m)


@st.cache_resource(show_spinner="Loading Whisper...")
def get_asr(m):
    from voicesent.asr import WhisperASR
    return WhisperASR(m)


try:
    sentiment = get_sentiment(model_id)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the sentiment model '{model_id}': {e}\n\n"
             "Train one with `python -m voicesent.train --data data/mosi.csv`, "
             "or enter a Hugging Face model id in the sidebar.")
    st.stop()

tab_voice, tab_text = st.tabs(["Speech", "Text"])
with tab_voice:
    rec = st.audio_input("Record a clip")
    up = st.file_uploader("…or upload audio", type=["wav", "flac", "ogg", "mp3"])
    src = rec or up
with tab_text:
    typed = st.text_area("Type or paste text to analyse without speech")


def show(overall, segments=None, transcript=None):
    if transcript is not None:
        st.subheader("Transcript")
        st.write(transcript)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment score", f"{overall.score:+.2f}", help="-3 (very negative) to +3 (very positive)")
    c2.metric("Polarity", overall.polarity)
    c3.metric("Intensity", overall.intensity)
    c4.metric("Confidence", f"{overall.confidence:.0%}", help=f"± {overall.std:.2f} across samples")
    st.progress((overall.score + 3) / 6)
    if len(overall.samples) > 1:
        hist, edges = np.histogram(overall.samples, bins=np.linspace(-3, 3, 25))
        st.caption("Score distribution across uncertainty samples")
        st.bar_chart(pd.DataFrame({"samples": hist}, index=np.round((edges[:-1] + edges[1:]) / 2, 2)))
    if segments:
        st.subheader("By segment")
        seg = pd.DataFrame(segments)
        st.line_chart(seg.set_index(seg.index + 1)["score"])
        cols = [c for c in ["start", "end", "text", "score", "polarity", "confidence"] if c in seg]
        st.dataframe(seg[cols].round(2), hide_index=True, width="stretch")


if src is not None:
    try:
        audio = load_audio(src.getvalue())
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read this audio file ({e}). Try WAV or FLAC.")
        st.stop()
    st.audio(src)
    if duration(audio) < 0.5:
        st.warning("The clip is shorter than half a second. Record a full sentence.")
        st.stop()
    with st.spinner("Transcribing and analysing..."):
        result = VoiceSentiment(get_asr(asr_id), sentiment)(audio, mc)
    if not result["transcript"]:
        st.warning("No speech was detected in this clip.")
    else:
        show(result["overall"], result["segments"], result["transcript"])
elif typed.strip():
    show(sentiment.predict([typed.strip()], mc)[0])
else:
    st.info("Record or upload a clip, or type some text, to see its sentiment.")
