"""Eksperimen FastText/LDA/BiLSTM sesuai proposal asistensi Malik.

LDA per musim hanya memakai TRAIN. Saat inferensi semua bank ditransformasi,
tanpa memilih bank berdasarkan label musim target (mencegah label leakage).
"""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import re

from gensim.models import FastText, Word2Vec
import joblib
import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from wisata.annotations import LABELS, validate_annotations


@dataclass
class Config:
    scenario: str = "fusion"
    embedding: str = "fasttext"
    seed: int = 42
    embedding_dim: int = 64
    hidden_dim: int = 32
    topics_per_season: int = 5
    max_tokens: int = 96
    batch_size: int = 32
    epochs: int = 10
    embedding_epochs: int = 10
    lda_iterations: int = 10
    learning_rate: float = 0.001
    subword_buckets: int = 50000


class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(StopWordRemoverFactory().get_stop_words()) - {"tidak", "bukan", "belum", "jangan", "kurang"}
        self.stemmer = StemmerFactory().create_stemmer()

    def __call__(self, text: str) -> list[str]:
        words = [word for word in re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
                 if word not in self.stopwords]
        return self.stemmer.stem(" ".join(words)).split() or ["kosong"]


class TopicBank:
    def __init__(self, config: Config):
        self.config = config
        self.vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                          max_features=20000)

    def fit(self, tokens: list[list[str]], seasons: np.ndarray):
        counts = self.vectorizer.fit_transform([" ".join(row) for row in tokens])
        self.seasons = sorted(set(seasons))
        self.models = []
        for season in self.seasons:
            model = LatentDirichletAllocation(n_components=self.config.topics_per_season,
                                              max_iter=self.config.lda_iterations,
                                              random_state=self.config.seed, n_jobs=1)
            model.fit(counts[seasons == season])
            self.models.append(model)
        return self

    def transform(self, tokens: list[list[str]]) -> np.ndarray:
        counts = self.vectorizer.transform([" ".join(row) for row in tokens])
        return np.hstack([model.transform(counts) for model in self.models]).astype(np.float32)


def fit_embedding(tokens: list[list[str]], config: Config):
    options = dict(sentences=tokens, vector_size=config.embedding_dim, window=5,
                   min_count=1, sg=1, workers=1, seed=config.seed, epochs=config.embedding_epochs)
    if config.embedding == "fasttext":
        return FastText(**options, min_n=3, max_n=6, bucket=config.subword_buckets)
    return Word2Vec(**options)


def vectorize(tokens: list[str], topics: np.ndarray, embedding, config: Config) -> torch.Tensor:
    if config.scenario == "lda":
        return torch.from_numpy(topics[None, :])
    # FastText dapat menyusun vektor OOV dari subword, tanpa fit ulang pada test.
    # Word2Vec baseline memakai vektor nol untuk kata di luar vocabulary train.
    vectors = np.stack([embedding.wv[word] if word in embedding.wv
                        else np.zeros(config.embedding_dim, dtype=np.float32)
                        for word in tokens[:config.max_tokens]])
    if config.scenario == "fusion":
        vectors = np.concatenate([vectors, np.repeat(topics[None, :], len(vectors), axis=0)], axis=1)
    return torch.from_numpy(vectors.astype(np.float32))


class ReviewDataset(Dataset):
    def __init__(self, tokens, topics, labels, embedding, config):
        self.tokens, self.topics, self.labels = tokens, topics, labels
        self.embedding, self.config = embedding, config

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return vectorize(self.tokens[i], self.topics[i], self.embedding, self.config), self.labels[i]


def collate(rows):
    features, labels = zip(*rows)
    return (pad_sequence(features, batch_first=True), torch.tensor([len(x) for x in features]),
            torch.tensor(np.stack(labels), dtype=torch.long))


class MultiAspectBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, class_counts: list[int]):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim * 2, count) for count in class_counts])

    def forward(self, features, lengths):
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        document = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return [head(document) for head in self.heads]


def split_annotations(df: pd.DataFrame, seed: int):
    # Teks identik lintas destinasi wajib tetap dalam split yang sama.
    groups = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                .split(df, groups=groups))


def run_experiment(df: pd.DataFrame, output: Path, config: Config) -> dict:
    if config.scenario not in {"fasttext", "lda", "fusion"} or config.embedding not in {"fasttext", "word2vec"}:
        raise ValueError("Skenario atau embedding tidak dikenal.")
    if any(getattr(config, name) < 1 for name in ("embedding_dim", "hidden_dim", "topics_per_season",
                                                "max_tokens", "batch_size", "epochs", "embedding_epochs",
                                                "lda_iterations", "subword_buckets")):
        raise ValueError("Parameter ukuran dan jumlah iterasi harus positif.")
    df = validate_annotations(df).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Minimal 20 anotasi approved untuk menjalankan pipeline eksperimen.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Folder output harus kosong agar hasil eksperimen lama tidak tertimpa.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train_ids, test_ids = split_annotations(df, config.seed)
    classes = {label: sorted(df.iloc[train_ids][label].unique()) for label in LABELS}
    for label in LABELS:
        if len(classes[label]) < 2:
            raise ValueError(f"Train memerlukan setidaknya dua kelas pada {label}.")
        if not set(df.iloc[test_ids][label]).issubset(classes[label]):
            raise ValueError(f"Test berisi kelas {label} yang tidak ada di train; revisi desain split/data.")

    processor = TextPreprocessor()
    tokens = [processor(text) for text in df["text"]]
    train_tokens = [tokens[i] for i in train_ids]
    embedding = None if config.scenario == "lda" else fit_embedding(train_tokens, config)
    bank = None
    topics = np.empty((len(df), 0), dtype=np.float32)
    if config.scenario != "fasttext":
        bank = TopicBank(config).fit(train_tokens, df.iloc[train_ids]["season_label"].to_numpy())
        topics = bank.transform(tokens)
    encoded = np.column_stack([df[label].map({c: i for i, c in enumerate(classes[label])}) for label in LABELS])
    input_dim = (config.embedding_dim if config.scenario != "lda" else 0) + topics.shape[1]
    model = MultiAspectBiLSTM(input_dim, config.hidden_dim, [len(classes[label]) for label in LABELS])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = lambda ids, shuffle: DataLoader(
        ReviewDataset([tokens[i] for i in ids], topics[ids], encoded[ids], embedding, config),
        batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0)
    train_loader, test_loader = loader(train_ids, True), loader(test_ids, False)
    losses = []
    for _ in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for features, lengths, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features, lengths)
            loss = sum(nn.functional.cross_entropy(value, labels[:, i]) for i, value in enumerate(outputs))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(labels)
        losses.append(epoch_loss / len(train_ids))

    model.eval()
    predictions = []
    with torch.inference_mode():
        for features, lengths, _ in test_loader:
            predictions.append(torch.stack([value.argmax(1) for value in model(features, lengths)], dim=1).numpy())
    pred = np.vstack(predictions)
    truth = encoded[test_ids]
    metrics = {label: {"micro_f1": float(f1_score(truth[:, i], pred[:, i], average="micro")),
                       "hamming_loss": float(np.mean(truth[:, i] != pred[:, i]))}
               for i, label in enumerate(LABELS)}
    metrics["overall_hamming_loss"] = float(np.mean(truth != pred))
    output.mkdir(parents=True, exist_ok=True)
    if embedding is not None:
        embedding.save(str(output / "embedding.gensim"))
    if bank is not None:
        joblib.dump(bank, output / "topics.joblib")
    torch.save(model.state_dict(), output / "classifier.pt")
    metadata = {"config": asdict(config), "classes": classes, "input_dim": input_dim,
                "train_review_ids": df.iloc[train_ids]["review_id"].tolist(),
                "test_review_ids": df.iloc[test_ids]["review_id"].tolist(),
                "train_rows": len(train_ids), "test_rows": len(test_ids),
                "annotation_sha256": sha256(df.to_csv(index=False).encode()).hexdigest(),
                "metrics": metrics, "training_losses": losses,
                "split": "grouped-by-normalized-review-text-80-20",
                "topic_inference": "concatenate-all-season-banks-no-target-label-routing"}
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model hanya dari anotasi approved.")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=["fasttext", "lda", "fusion"], default="fusion")
    parser.add_argument("--embedding", choices=["fasttext", "word2vec"], default="fasttext")
    parser.add_argument("--region", default="madura", help="Filter wilayah anotasi; gunakan 'all' hanya untuk eksperimen tambahan.")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    df = pd.read_csv(args.annotations)
    if "region" not in df:
        parser.error("Kolom region diperlukan untuk seleksi data Madura sesuai proposal.")
    if args.region != "all":
        df = df.loc[df["region"].astype(str).str.lower().eq(args.region.lower())]
    config = Config(scenario=args.scenario, embedding=args.embedding, epochs=args.epochs)
    try:
        result = run_experiment(df, args.output, config)
    except ValueError as exc:
        parser.exit(1, f"Training belum dapat dijalankan: {exc}\n")
    print(json.dumps(result["metrics"], indent=2))
