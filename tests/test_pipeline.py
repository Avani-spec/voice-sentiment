import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voicesent.asr import Segment
from voicesent.audio import SR, load_audio
from voicesent.data import load_splits
from voicesent.predict import SentimentModel, VoiceSentiment, split_sentences
from voicesent.train import main as train_main


def test_load_audio_resamples_and_downmixes():
    t = np.linspace(0, 1, 44_100, endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.zeros_like(t)], 1).astype(np.float32)
    buf = io.BytesIO(); sf.write(buf, stereo, 44_100, format="WAV")
    x = load_audio(buf.getvalue())
    assert x.ndim == 1 and abs(len(x) - SR) <= 1 and x.dtype == np.float32


def test_group_split_is_speaker_disjoint(toy_csv, tmp_path):
    import pandas as pd
    df = pd.read_csv(toy_csv).drop(columns="split")
    p = tmp_path / "nosplit.csv"; df.to_csv(p, index=False)
    s = load_splits(str(p))
    vids = [set(s[k].video_id) for k in ("train", "valid", "test")]
    assert not (vids[0] & vids[1]) and not (vids[0] & vids[2]) and not (vids[1] & vids[2])
    assert sum(len(v) for v in s.values()) == len(df)


def test_train_learns_and_predict(tiny_model_dir, toy_csv, tmp_path):
    out = tmp_path / "run"
    test = train_main(["--data", toy_csv, "--model", tiny_model_dir, "--out", str(out),
                       "--epochs", "25", "--lr", "3e-3", "--batch-size", "8", "--patience", "25", "--warmup", "0.05"])
    assert test["acc2_nonzero"] >= 0.9                   # toy task is easy
    assert json.loads((out / "metrics.json").read_text())["test"]["mae"] == test["mae"]

    sm = SentimentModel(str(out / "best"))
    pos, neg = sm.predict(["i loved this movie", "i hated this movie"], mc_samples=20)
    assert pos.score > 0 > neg.score
    assert pos.polarity == "positive" and neg.polarity == "negative"
    assert 0 <= pos.confidence <= 1 and len(pos.samples) == 20 and pos.std >= 0

    class FakeASR:  # stands in for Whisper
        def __call__(self, audio):
            return ("I loved this movie. I hated this movie.",
                    [Segment(0, 1.5, "I loved this movie."), Segment(1.5, 3, "I hated this movie.")])

    r = VoiceSentiment(FakeASR(), sm)(np.zeros(SR * 3, dtype=np.float32), mc_samples=5)
    assert [s["polarity"] for s in r["segments"]] == ["positive", "negative"]


def test_split_sentences():
    assert split_sentences("Great film! I mean it. Ok") == ["Great film!", "I mean it."]


def test_three_class_classifier_mapping(tiny_model_dir, tmp_path):
    from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
    cfg = RobertaConfig.from_pretrained(tiny_model_dir)
    cfg.num_labels, cfg.problem_type = 3, "single_label_classification"
    cfg.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    cfg.label2id = {v: k for k, v in cfg.id2label.items()}
    d = tmp_path / "cls"
    RobertaForSequenceClassification(cfg).save_pretrained(d)
    AutoTokenizer.from_pretrained(tiny_model_dir).save_pretrained(d)
    sm = SentimentModel(str(d))
    assert not sm.regression and (sm.neg, sm.pos) == (0, 2)
    p = sm.predict(["really enjoyed it"], mc_samples=3)[0]
    assert -3 <= p.score <= 3
