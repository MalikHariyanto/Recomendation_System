"""Muat artefak classifier lokal yang dibuat pipeline proyek ini."""

import json
from pathlib import Path

from gensim.models import FastText, Word2Vec
import joblib
import numpy as np
import torch

from wisata.annotations import LABELS
from wisata.training import Config, MultiAspectBiLSTM, TextPreprocessor, token_cache_key, vectorize


class Classifier:
    def __init__(self, directory: Path):
        self.directory = directory
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

    def predict_many(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        from torch.nn.utils.rnn import pad_sequence
        if batch_size < 1 or any(not text.strip() for text in texts):
            raise ValueError("Batch harus positif dan teks tidak boleh kosong.")
        if not texts:
            return []
        cache = self.directory.parent / ("tokens-" + token_cache_key(texts) + ".json")
        cached_tokens = json.loads(cache.read_text()) if cache.exists() else None
        if cached_tokens is not None and (len(cached_tokens) != len(texts) or any(not row for row in cached_tokens)):
            raise ValueError("Cache preprocessing tidak sesuai dengan korpus.")
        result = []
        for start in range(0, len(texts), batch_size):
            tokens = (cached_tokens[start:start + batch_size] if cached_tokens is not None else
                      [self.processor(text) for text in texts[start:start + batch_size]])
            topics = (np.empty((len(tokens), 0), dtype=np.float32) if self.bank is None
                      else self.bank.transform(tokens))
            sequences = [vectorize(row, topic, self.embedding, self.config) for row, topic in zip(tokens, topics)]
            with torch.inference_mode():
                logits = self.model(pad_sequence(sequences, batch_first=True),
                                    torch.tensor([len(row) for row in sequences]))
            for i in range(len(tokens)):
                result.append({label: self.manifest["classes"][label][int(values.argmax(1)[i])]
                               for label, values in zip(LABELS, logits)})
        return result
