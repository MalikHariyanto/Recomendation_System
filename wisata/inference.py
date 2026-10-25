"""Muat artefak classifier lokal yang dibuat pipeline proyek ini."""

import json
from pathlib import Path

from gensim.models import FastText, Word2Vec
import joblib
import numpy as np
import torch

from wisata.annotations import LABELS
from wisata.training import Config, MultiAspectBiLSTM, TextPreprocessor, vectorize


class Classifier:
    def __init__(self, directory: Path):
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.config = Config(**self.manifest["config"])
        self.processor = TextPreprocessor()
        self.bank = None if self.config.scenario == "fasttext" else joblib.load(directory / "topics.joblib")
        self.embedding = None
        if self.config.scenario != "lda":
            cls = FastText if self.config.embedding == "fasttext" else Word2Vec
            self.embedding = cls.load(str(directory / "embedding.gensim"))
        self.model = MultiAspectBiLSTM(self.manifest["input_dim"], self.config.hidden_dim,
                                       [len(self.manifest["classes"][label]) for label in LABELS])
        self.model.load_state_dict(torch.load(directory / "classifier.pt", map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("Teks ulasan tidak boleh kosong.")
        tokens = self.processor(text)
        topics = np.empty(0, dtype=np.float32) if self.bank is None else self.bank.transform([tokens])[0]
        features = vectorize(tokens, topics, self.embedding, self.config)
        with torch.inference_mode():
            logits = self.model(features[None, :, :], torch.tensor([len(features)]))
        return {label: self.manifest["classes"][label][int(values.argmax(1)[0])]
                for label, values in zip(LABELS, logits)}
