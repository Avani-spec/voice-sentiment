"""Speech-to-text with OpenAI Whisper via Hugging Face transformers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SR


@dataclass
class Segment:
    start: float
    end: float
    text: str


class WhisperASR:
    def __init__(self, model: str = "openai/whisper-small", device: int | str | None = None, language: str | None = "en"):
        import torch
        from transformers import pipeline
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device,
                             chunk_length_s=30)
        self.language = language

    def __call__(self, audio: np.ndarray) -> tuple[str, list[Segment]]:
        kw = {"return_timestamps": True}
        if self.language:
            kw["generate_kwargs"] = {"language": self.language, "task": "transcribe"}
        out = self.pipe({"raw": audio, "sampling_rate": SR}, **kw)
        segs = []
        for c in out.get("chunks", []):
            s, e = c.get("timestamp") or (None, None)
            if c["text"].strip():
                segs.append(Segment(float(s or 0.0), float(e if e is not None else len(audio) / SR), c["text"].strip()))
        return out["text"].strip(), segs
