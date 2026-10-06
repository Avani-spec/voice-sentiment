"""Fine-tune RoBERTa as a sentiment regressor (score in [-3, 3]).

    python -m voicesent.train --data data/mosi.csv --model roberta-base --out runs/roberta-mosi
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import TextRegression, load_splits, make_collate
from .metrics import mosi_metrics


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys = [], []
    for b in loader:
        y = b.pop("labels")
        out = model(**{k: v.to(device) for k, v in b.items()}).logits.squeeze(-1)
        preds.append(out.clamp(-3, 3).cpu()); ys.append(y)
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--out", default="runs/roberta-mosi")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--patience", type=int, default=3, help="early stopping on validation MAE")
    p.add_argument("--group-split", action="store_true", help="ignore the split column; split by video_id")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(a.data, a.group_split, a.seed)
    print({k: len(v) for k, v in splits.items()})
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=1, problem_type="regression", ignore_mismatched_sizes=True).to(device)

    collate = make_collate(tok)
    dl = {s: DataLoader(TextRegression(df, tok, a.max_len), batch_size=a.batch_size,
                        shuffle=(s == "train"), collate_fn=collate) for s, df in splits.items()}
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": a.weight_decay},
              {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr)
    total = a.epochs * len(dl["train"])
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    loss_fn = torch.nn.L1Loss()  # MAE objective; L1 is the standard choice on MOSI

    best, bad, history = math.inf, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for b in dl["train"]:
            y = b.pop("labels").to(device)
            pred = model(**{k: v.to(device) for k, v in b.items()}).logits.squeeze(-1)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        val = mosi_metrics(*predict(model, dl["valid"], device))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()},
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if val["mae"] < best - 1e-4:
            best, bad = val["mae"], 0
            model.save_pretrained(out / "best"); tok.save_pretrained(out / "best")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    model = AutoModelForSequenceClassification.from_pretrained(out / "best").to(device)
    test = mosi_metrics(*predict(model, dl["test"], device))
    print("TEST", json.dumps({k: round(v, 4) for k, v in test.items()}))
    (out / "metrics.json").write_text(json.dumps({"config": vars(a), "history": history, "test": test}, indent=2))
    return test


if __name__ == "__main__":
    main()
