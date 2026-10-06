"""Dataset loading.

The training CSV has one row per utterance with columns
``text, label, split`` and optionally ``video_id`` / ``segment_id``.
``scripts/prepare_mosi.py`` writes this file for CMU-MOSI using the standard
folds. Any other dataset in the same format works too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_splits(path: str, group_split: bool = False, seed: int = 0):
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["text"] = df.text.astype(str).str.strip()
    if "split" not in df or group_split:
        # group-aware split: every utterance from one video/speaker lands in
        # the same split, so the test set has only unseen speakers
        from sklearn.model_selection import GroupShuffleSplit
        groups = df.get("video_id", pd.Series(np.arange(len(df))))
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(df, groups=groups))
        rest = df.iloc[te]
        va, te2 = next(GroupShuffleSplit(1, test_size=0.7, random_state=seed).split(rest, groups=groups.iloc[te]))
        df["split"] = "train"
        df.loc[rest.index[va], "split"] = "valid"
        df.loc[rest.index[te2], "split"] = "test"
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test")}


class TextRegression(Dataset):
    def __init__(self, df, tokenizer, max_len: int = 128):
        self.enc = tokenizer(df.text.tolist(), truncation=True, max_length=max_len)
        self.y = df.label.astype(float).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


def make_collate(tokenizer):
    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch], dtype=torch.float)
        out = tokenizer.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out
    return collate
