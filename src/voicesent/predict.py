"""Sentiment inference with uncertainty.

Works with two kinds of model directories / Hub ids:

* the regression model produced by ``voicesent.train`` (one output in [-3, 3]);
* any 3-class sentiment classifier (negative / neutral / positive), whose
  output is mapped to a score as 3 * (P(positive) - P(negative)). Handy as a
  demo before you have trained your own model.

Confidence comes from Monte-Carlo dropout: the model is run ``mc_samples``
times with dropout active; we report the mean score, its standard deviation
and the share of samples agreeing on the polarity.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .metrics import intensity_label


@dataclass
class Prediction:
    text: str
    score: float           # -3 .. 3
    std: float
    polarity: str          # positive / negative / neutral
    intensity: str         # 7-class label
    confidence: float      # share of MC samples agreeing with the polarity
    samples: list

    def as_dict(self):
        d = asdict(self); d.pop("samples"); return d


class SentimentModel:
    def __init__(self, model: str, device: str | None = None, max_len: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device)
        self.max_len = max_len
        cfg = self.model.config
        self.regression = cfg.num_labels == 1
        if not self.regression:
            labels = {i: str(l).lower() for i, l in cfg.id2label.items()}
            find = lambda key: next((i for i, l in labels.items() if key in l), None)  # noqa: E731
            self.pos, self.neg = find("pos"), find("neg")
            if self.pos is None or self.neg is None:
                raise ValueError(f"cannot find positive/negative labels in {labels}")

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.regression:
            return logits.squeeze(-1).clamp(-3, 3)
        p = logits.softmax(-1)
        return 3 * (p[:, self.pos] - p[:, self.neg])

    @torch.no_grad()
    def predict(self, texts: list[str], mc_samples: int = 30, neutral_band: float = 0.5) -> list[Prediction]:
        if not texts:
            return []
        enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
        self.model.eval()
        point = self._scores(self.model(**enc).logits).cpu().numpy()
        if mc_samples > 1:
            self.model.train()  # enable dropout
            S = torch.stack([self._scores(self.model(**enc).logits) for _ in range(mc_samples)], 1).cpu().numpy()
            self.model.eval()
        else:
            S = point[:, None]
        out = []
        for t, p, s in zip(texts, point, S):
            pol = "neutral" if abs(p) < neutral_band else ("positive" if p > 0 else "negative")
            agree = np.mean(np.abs(s) < neutral_band) if pol == "neutral" else np.mean(np.sign(s) == np.sign(p))
            out.append(Prediction(t, float(p), float(s.std()), pol, intensity_label(p), float(agree), s.tolist()))
        return out


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p.split()) >= 2]


class VoiceSentiment:
    """audio -> transcript -> overall + per-segment sentiment."""

    def __init__(self, asr, sentiment: SentimentModel):
        self.asr, self.sentiment = asr, sentiment

    def __call__(self, audio: np.ndarray, mc_samples: int = 30) -> dict:
        text, segments = self.asr(audio)
        if not text:
            return {"transcript": "", "overall": None, "segments": []}
        overall = self.sentiment.predict([text], mc_samples)[0]
        units = [(s.start, s.end, s.text) for s in segments] or [(None, None, t) for t in split_sentences(text)]
        preds = self.sentiment.predict([u[2] for u in units], mc_samples) if len(units) > 1 else []
        seg_rows = [{"start": u[0], "end": u[1], **p.as_dict()} for u, p in zip(units, preds)]
        return {"transcript": text, "overall": overall, "segments": seg_rows}
