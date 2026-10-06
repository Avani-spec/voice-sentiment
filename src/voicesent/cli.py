"""Predict sentiment for audio files or text.

    python -m voicesent.cli audio clip.wav --model runs/roberta-mosi/best
    python -m voicesent.cli text "I really loved this movie" --model runs/roberta-mosi/best
"""
import argparse
import json

from .predict import SentimentModel, VoiceSentiment


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["audio", "text"])
    p.add_argument("inputs", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--asr", default="openai/whisper-small")
    p.add_argument("--mc", type=int, default=30)
    a = p.parse_args(argv)
    sm = SentimentModel(a.model)
    if a.kind == "text":
        for r in sm.predict(a.inputs, a.mc):
            print(json.dumps(r.as_dict()))
        return
    from .asr import WhisperASR
    from .audio import load_audio
    vs = VoiceSentiment(WhisperASR(a.asr), sm)
    for f in a.inputs:
        r = vs(load_audio(f), a.mc)
        print(json.dumps({"file": f, "transcript": r["transcript"],
                          "overall": r["overall"].as_dict() if r["overall"] else None,
                          "segments": r["segments"]}, indent=2))


if __name__ == "__main__":
    main()
