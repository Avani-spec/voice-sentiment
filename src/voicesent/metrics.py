"""Standard CMU-MOSI evaluation metrics.

Labels are continuous sentiment scores in [-3, 3].

* MAE, Pearson correlation
* Acc-2 / F1 ("non-zero"): positive vs negative, excluding neutral (== 0) labels
* Acc-2 / F1 ("has-0"):   label >= 0 vs label < 0, all utterances
* Acc-7: scores rounded and clipped to the 7 integer classes -3..3
F1 is the weighted F1, as in most MOSI papers.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def mosi_metrics(pred, true) -> dict:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = {"mae": float(np.abs(pred - true).mean()),
           "corr": float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 1 and pred.std() > 0 else 0.0}
    nz = true != 0
    if nz.any():
        out["acc2_nonzero"] = float(accuracy_score(true[nz] > 0, pred[nz] > 0))
        out["f1_nonzero"] = float(f1_score(true[nz] > 0, pred[nz] > 0, average="weighted"))
    out["acc2_has0"] = float(accuracy_score(true >= 0, pred >= 0))
    out["f1_has0"] = float(f1_score(true >= 0, pred >= 0, average="weighted"))
    out["acc7"] = float(accuracy_score(np.clip(np.round(true), -3, 3), np.clip(np.round(pred), -3, 3)))
    return out


INTENSITY = {-3: "strongly negative", -2: "negative", -1: "weakly negative", 0: "neutral",
             1: "weakly positive", 2: "positive", 3: "strongly positive"}


def intensity_label(score: float) -> str:
    return INTENSITY[int(np.clip(np.round(score), -3, 3))]
