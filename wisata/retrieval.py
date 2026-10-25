"""Retrieval BGE-M3 dense: cosine ulasan, rata-rata top-m per destinasi.

Encoder asli wajib tersedia. Tidak ada fallback TF-IDF atau skor acak untuk UI.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from wisata.annotations import validate_annotations


MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class BGEEncoder:
    model_id = MODEL_ID
    revision = MODEL_REVISION

    def __init__(self, max_length: int = 256, precision: str = "float32"):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(4, len(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else 4)
        self.max_length = max_length
        if precision not in {"float32", "bfloat16"}:
            raise ValueError("Precision encoder tidak dikenal.")
        self.precision = precision
        self.model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu",
                                         cache_folder=os.environ.get("HF_HOME"), trust_remote_code=False,
                                         model_kwargs={"use_safetensors": False,
                                                       "dtype": getattr(torch, precision)})
        self.model.max_seq_length = max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=16, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 32).astype(np.float32)


def normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding harus matriks finite.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms < 1e-10).any():
        raise ValueError("Encoder menghasilkan vektor nol.")
    return vectors / norms


def build_index(labels: pd.DataFrame, directory: Path, encoder, classifier=None) -> dict:
    labels = validate_annotations(labels).reset_index(drop=True)
    if not labels["region"].eq("madura").all():
        raise ValueError("Indeks utama hanya menerima cakupan Madura.")
    required = {"destinasi", "source_rating"}
    if not required.issubset(labels):
        raise ValueError("Indeks memerlukan nama destinasi dan rating sumber.")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Folder indeks harus baru/kosong agar artefak lama tidak tertimpa.")
    reviews = labels.copy()
    sentiment_source = "rating-derived"
    if classifier is not None:
        predictions = pd.DataFrame(classifier.predict_many(reviews["text"].tolist())).add_prefix("predicted_")
        reviews = pd.concat([reviews, predictions], axis=1)
        sentiment_source = "bilstm-prediction"
    vectors = normalized(encoder.encode(reviews["text"].tolist()))
    if len(vectors) != len(reviews):
        raise ValueError("Jumlah embedding tidak sesuai urutan ulasan.")
    directory.mkdir(parents=True, exist_ok=True)
    reviews.to_json(directory / "reviews.json", orient="records", force_ascii=False)
    np.save(directory / "embeddings.npy", vectors, allow_pickle=False)
    manifest = {"version": 1, "model_id": encoder.model_id, "revision": encoder.revision,
                "precision": getattr(encoder, "precision", "float32"),
                "max_length": encoder.max_length, "reviews": len(reviews),
                "destinations": reviews["destinasi"].nunique(),
                "corpus_sha256": sha256(json.dumps(reviews["text"].tolist(), ensure_ascii=False).encode()).hexdigest(),
                "sentiment_source": sentiment_source,
                "files": {name: sha256((directory / name).read_bytes()).hexdigest()
                          for name in ["reviews.json", "embeddings.npy"]}}
    if classifier is not None:
        manifest["classifier_annotation_sha256"] = classifier.manifest["annotation_sha256"]
        manifest["classifier_config"] = classifier.manifest["config"]
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class ReviewIndex:
    def __init__(self, directory: Path, encoder=None):
        self.manifest = json.loads((directory / "manifest.json").read_text())
        for name in ["reviews.json", "embeddings.npy"]:
            if sha256((directory / name).read_bytes()).hexdigest() != self.manifest["files"][name]:
                raise ValueError("Checksum indeks tidak cocok; bangun ulang dari sumber tepercaya.")
        self.reviews = pd.read_json(directory / "reviews.json", orient="records")
        self.vectors = normalized(np.load(directory / "embeddings.npy", allow_pickle=False))
        if len(self.reviews) != len(self.vectors):
            raise ValueError("Indeks dan metadata ulasan tidak selaras.")
        if encoder is None:
            if self.manifest["model_id"] != MODEL_ID or self.manifest["revision"] != MODEL_REVISION:
                raise ValueError("Artefak bukan indeks BGE-M3 dari revisi yang didukung.")
        if encoder is not None and (encoder.model_id, encoder.revision, encoder.max_length) != (
            self.manifest["model_id"], self.manifest["revision"], self.manifest["max_length"]):
            raise ValueError("Encoder query dan encoder indeks harus identik.")
        if encoder is not None and getattr(encoder, "precision", "float32") != self.manifest.get("precision", "float32"):
            raise ValueError("Precision encoder query dan indeks harus identik.")
        self.encoder = encoder

    def search(self, query: str, season: str = "Semua", category: str = "Semua",
               sentiment: str = "Semua", k: int = 5, top_m: int = 5) -> pd.DataFrame:
        if not query.strip() or k < 1 or top_m < 1:
            raise ValueError("Query wajib diisi dan k/top_m harus positif.")
        if self.encoder is None:
            self.encoder = BGEEncoder(max_length=self.manifest["max_length"], precision=self.manifest.get("precision", "float32"))
        vector = self.encoder.encode([query])[0]
        return self.search_vector(vector, season, category, sentiment, k, top_m)

    def search_vector(self, vector: np.ndarray, season: str = "Semua", category: str = "Semua",
                      sentiment: str = "Semua", k: int = 5, top_m: int = 5,
                      exclude_review_id: str | None = None) -> pd.DataFrame:
        """Untuk evaluasi known-item, embedding query yang sama dapat dipakai ulang."""
        if k < 1 or top_m < 1:
            raise ValueError("k/top_m harus positif.")
        if season not in {"Semua", "Hujan", "Kemarau"}:
            raise ValueError("Musim pencarian tidak dikenal.")
        mask = np.ones(len(self.reviews), dtype=bool)
        if exclude_review_id is not None:
            mask &= self.reviews["review_id"].ne(exclude_review_id).to_numpy()
        if season != "Semua":
            word = "Penghujan" if season == "Hujan" else "Kemarau"
            mask &= self.reviews["season_label"].str.contains(word, regex=False).to_numpy()
        if category != "Semua":
            mask &= self.reviews["tourism_label"].eq(category).to_numpy()
        sentiment_column = ("predicted_sentiment_label" if self.manifest["sentiment_source"] == "bilstm-prediction"
                            else "sentiment_label")
        if sentiment != "Semua":
            if sentiment not in {"positif", "negatif", "netral"}:
                raise ValueError("Sentimen tidak dikenal.")
            mask &= self.reviews[sentiment_column].eq(sentiment).to_numpy()
        if not mask.any():
            return pd.DataFrame()
        q = normalized(np.asarray(vector)[None, :])
        scores = self.vectors[mask] @ q[0]
        candidates = self.reviews.loc[mask].copy().assign(cosine=scores)
        rows = []
        for name, group in candidates.groupby("destinasi", sort=True):
            ordered = group.sort_values("cosine", ascending=False, kind="stable")
            first = ordered.iloc[0]
            rows.append({"destinasi": name, "wisata": first["tourism_label"], "musim": first["season_label"],
                         "skor": float(ordered.head(top_m)["cosine"].mean()),
                         "rating_mean": float(group["source_rating"].mean()),
                         "ulasan_bukti": first["text"], "sentimen_bukti": first[sentiment_column],
                         "sumber_sentimen": self.manifest["sentiment_source"],
                         "ulasan_kandidat": len(group)})
        return pd.DataFrame(rows).sort_values("skor", ascending=False, kind="stable").head(k).reset_index(drop=True)


def evaluate_known_items(index: ReviewIndex, k: int = 5, seed: int = 42) -> dict:
    """Satu query ulasan per destinasi, dikeluarkan dari kandidat; bukan human qrels."""
    rng = np.random.default_rng(seed)
    records = []
    for destination, group in index.reviews.groupby("destinasi", sort=True):
        if len(group) < 2:
            continue
        position = int(rng.choice(group.index))
        row = index.reviews.iloc[position]
        ranking = index.search_vector(index.vectors[position], k=k, exclude_review_id=row["review_id"])
        names = ranking["destinasi"].tolist() if not ranking.empty else []
        rank = names.index(destination) + 1 if destination in names else None
        records.append({"review_id": row["review_id"], "destination": destination, "rank": rank,
                        "precision": (1.0 / len(names) if rank is not None and names else 0.),
                        "recall": float(rank is not None), "reciprocal_rank": 0. if rank is None else 1. / rank})
    if not records:
        raise ValueError("Tidak ada destinasi dengan dua ulasan untuk known-item.")
    return {"protocol": "one-review-per-destination-leave-query-out-in-corpus", "k": k,
            "queries": len(records), "seed": seed,
            "precision_at_k": float(np.mean([row["precision"] for row in records])),
            "recall_at_k": float(np.mean([row["recall"] for row in records])),
            "mrr_at_k": float(np.mean([row["reciprocal_rank"] for row in records])), "per_query": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    classifier = None
    if args.classifier:
        from wisata.inference import Classifier
        classifier = Classifier(args.classifier)
    # Baca label sebelum mengunduh encoder besar agar kesalahan data gagal lebih awal.
    labels = validate_annotations(pd.read_csv(args.labels))
    try:
        encoder = BGEEncoder(precision=args.precision)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, "Encoder BGE belum dapat dimuat (" + type(exc).__name__ + "). "
                    "Periksa akses huggingface.co dan us.aws.cdn.hf.co; lihat kebutuhan jaringan pada README.\n")
    result = build_index(labels, args.output, encoder, classifier=classifier)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
