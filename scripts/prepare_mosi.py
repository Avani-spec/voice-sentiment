"""Build data/mosi.csv (text, label, split, video_id, segment_id) from CMU-MOSI.

Uses the official CMU Multimodal SDK (pip install
git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK), which downloads
the word-level transcripts and opinion-segment labels and provides the
standard train / valid / test folds. Those folds are split by video, so
evaluation is on speakers not seen in training.

    python scripts/prepare_mosi.py --cache cmumosi/ --out data/mosi.csv

If the SDK download servers are unavailable, any CSV with the same columns
can be used instead; ``voicesent.train --group-split`` will then create a
video-disjoint split itself.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cmumosi/")
ap.add_argument("--out", default="data/mosi.csv")
a = ap.parse_args()

from mmsdk import mmdatasdk as md  # noqa: E402

recipe = {"words": md.cmu_mosi.raw["words"],
          "labels": md.cmu_mosi.labels["Opinion Segment Labels"]}
ds = md.mmdataset(recipe, a.cache)
ds.align("labels")  # cut the word stream into the labelled opinion segments -> keys "videoid[i]"

folds = md.cmu_mosi.standard_folds
split_of = {**{v: "train" for v in folds.standard_train_fold},
            **{v: "valid" for v in folds.standard_valid_fold},
            **{v: "test" for v in folds.standard_test_fold}}

rows = []
for key in ds["labels"].keys():
    vid = key.split("[")[0]
    if key not in ds["words"].keys() or vid not in split_of:
        continue
    words = [w[0].decode() if isinstance(w[0], bytes) else str(w[0]) for w in ds["words"][key]["features"]]
    text = " ".join(w for w in words if w and w != "sp").strip()
    if not text:
        continue
    label = float(np.asarray(ds["labels"][key]["features"]).reshape(-1)[0])
    rows.append({"segment_id": key, "video_id": vid, "split": split_of[vid], "text": text, "label": label})

df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(df.groupby("split").agg(utterances=("text", "size"), videos=("video_id", "nunique")))
print(f"wrote {len(df)} utterances to {a.out}")
