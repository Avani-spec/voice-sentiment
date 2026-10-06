# Voice sentiment analysis (Whisper + RoBERTa)

Speak or upload a clip and get its sentiment: a score from −3 (very negative)
to +3 (very positive), its polarity and intensity, and how confident the
model is.

```
audio ─► Whisper (speech-to-text, timestamps) ─► RoBERTa regressor ─► score per clip and per segment
                                                     └─ Monte-Carlo dropout ─► confidence, score distribution
```

## Model

- **Speech-to-text.** OpenAI Whisper through Hugging Face `transformers`
  (base, small or medium), with 30 s chunking and segment timestamps.
- **Sentiment.** `roberta-base` fine-tuned as a **regressor** on CMU-MOSI's
  continuous labels (−3…3). One model gives both polarity (sign) and
  intensity (magnitude).
  - Loss: L1 (MAE).
  - Optimiser: AdamW with weight decay (bias and LayerNorm excluded), linear
    warmup and decay, gradient clipping.
  - Early stopping on validation MAE.
- **Confidence.** At inference the model runs N times with dropout active
  (Monte-Carlo dropout). The app reports the mean score, its spread, and the
  share of samples that agree on the polarity.

## Data: CMU-MOSI

CMU-MOSI has 2,199 opinion utterances from YouTube movie reviews, each
labelled −3…3. `scripts/prepare_mosi.py` uses the official CMU Multimodal
SDK to build `data/mosi.csv` with the **standard folds**. Those folds are
split by video, so the test set contains only speakers not seen in training.
For other datasets in the same CSV format, `--group-split` makes a
video/speaker-disjoint split automatically.

## Usage

```bash
pip install -r requirements.txt
pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
python scripts/prepare_mosi.py --out data/mosi.csv

PYTHONPATH=src python -m voicesent.train --data data/mosi.csv --model roberta-base --out runs/roberta-mosi
PYTHONPATH=src python -m voicesent.cli text "I really loved this movie" --model runs/roberta-mosi/best
PYTHONPATH=src python -m voicesent.cli audio clip.wav --model runs/roberta-mosi/best
streamlit run app.py
```

To try the app before training, enter a public 3-class sentiment model such
as `cardiffnlp/twitter-roberta-base-sentiment-latest` in the sidebar. Its
probabilities are mapped to a score as 3·(P(positive) − P(negative)).

## Results

`voicesent.train` writes `metrics.json` with the standard MOSI metrics on the
test fold:

- MAE and Pearson correlation
- Acc-2 and F1, both non-zero and has-0 variants
- Acc-7

<!-- Fill in after training, e.g.
| Model | Acc-2 (non-0) | F1 (non-0) | MAE | Corr | Acc-7 |
|---|---|---|---|---|---|
| roberta-base, text only | | | | | |
-->

## Tests

`pytest -q` runs 9 tests with no downloads. A tiny randomly initialised
RoBERTa is built locally and fine-tuned on a toy dataset. This exercises:

- training, checkpointing and the metrics on a hand-computed example
- MC-dropout inference and the 3-class mapping
- audio resampling
- the speaker-disjoint split
- the speech pipeline, with a stand-in for Whisper
- the Streamlit app

## Layout

```
src/voicesent/
  audio.py    decode any audio to 16 kHz mono
  asr.py      Whisper wrapper (segments + timestamps)
  data.py     CSV loading, speaker-disjoint splits
  train.py    fine-tuning loop with early stopping
  metrics.py  MOSI metrics, intensity labels
  predict.py  MC-dropout inference, audio → per-segment sentiment
  cli.py
scripts/prepare_mosi.py
app.py        Streamlit (microphone, upload, or text)
```
