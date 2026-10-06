import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

POS = ["i loved this movie", "great acting and a wonderful story", "really enjoyed it", "fantastic fun"]
NEG = ["i hated this movie", "terrible acting and a boring story", "really disliked it", "awful waste"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A randomly initialised mini RoBERTa + tokenizer, saved locally (no downloads)."""
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

    d = tmp_path_factory.mktemp("tiny")
    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator([*POS, *NEG, ". !"], trainers.WordLevelTrainer(special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>", mask_token="<mask>", cls_token="<s>", sep_token="</s>")
    cfg = RobertaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=64, num_labels=1, problem_type="regression",
                        pad_token_id=tok.pad_token_id, hidden_dropout_prob=0.2)
    torch.manual_seed(0)  # deterministic initial weights
    RobertaForSequenceClassification(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture
def toy_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for v in range(30):
        for j in range(4):
            pos = rng.random() < 0.5
            text = POS[j] if pos else NEG[j]
            rows.append({"video_id": f"vid{v}", "text": text, "label": float(rng.uniform(1, 3) if pos else -rng.uniform(1, 3))})
    df = pd.DataFrame(rows)
    df["split"] = np.where(df.video_id.isin([f"vid{i}" for i in range(20)]), "train",
                           np.where(df.video_id.isin([f"vid{i}" for i in range(20, 25)]), "valid", "test"))
    p = tmp_path / "toy.csv"
    df.to_csv(p, index=False)
    return str(p)
