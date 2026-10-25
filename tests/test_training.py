"""Uji teknis dengan data sintetis, bukan benchmark kualitas penelitian."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("gensim")
pytest.importorskip("Sastrawi")

from wisata.annotations import LABELS
from wisata.inference import Classifier
from wisata.training import Config, TextPreprocessor, fit_embedding, run_experiment, split_annotations, token_cache_key


@pytest.fixture
def labeled_reviews():
    return pd.DataFrame([
        {"review_id": f"synthetic-{i}",
         "text": f"Wisata pantai bersih nyaman keluarga cerah contoh ulasan nomor {i}",
         "season_label": ["Kemarau", "Kemarau & Penghujan"][i % 2],
         "tourism_label": ["pantai", "sejarah", "taman", "religi", "lainnya"][i % 5],
         "sentiment_label": ["positif", "negatif", "netral"][i % 3],
         "region": "madura", "annotation_status": "approved"}
        for i in range(80)
    ])


@pytest.mark.parametrize("scenario,embedding", [
    ("fasttext", "fasttext"), ("lda", "fasttext"),
    ("fusion", "fasttext"), ("fusion", "word2vec"),
])
def test_synthetic_train_export_reload_predict(labeled_reviews, tmp_path, scenario, embedding):
    config = Config(scenario=scenario, embedding=embedding, embedding_dim=8, hidden_dim=4,
                    topics_per_season=2, epochs=1, embedding_epochs=1,
                    lda_iterations=1, subword_buckets=1000)
    directory = tmp_path / f"{scenario}-{embedding}"
    result = run_experiment(labeled_reviews, directory, config)
    assert result["train_rows"] == 64 and result["test_rows"] == 16
    assert not set(result["train_review_ids"]) & set(result["test_review_ids"])
    assert np.isfinite(result["training_losses"]).all()
    assert 0 <= result["metrics"]["overall_hamming_loss"] <= 1
    classifier = Classifier(directory)
    assert set(classifier.predict("pantai indah untuk keluarga")) == set(LABELS)
    assert classifier.predict_many(["pantai indah untuk keluarga"])[0] == classifier.predict("pantai indah untuk keluarga")
    text = "pantai indah untuk keluarga"
    expected = classifier.predict(text)
    cache = directory.parent / ("tokens-" + token_cache_key([text]) + ".json")
    cache.write_text(json.dumps([classifier.processor(text)]))
    classifier.processor = lambda text: (_ for _ in ()).throw(AssertionError("Should reuse matching token cache"))
    assert classifier.predict_many([text]) == [expected]
    with pytest.raises(ValueError, match="kosong"):
        classifier.predict("  ")
    assert json.loads((directory / "manifest.json").read_text())["config"]["scenario"] == scenario


def test_group_split_keeps_identical_review_text_together(labeled_reviews):
    labeled_reviews.loc[1, "text"] = labeled_reviews.loc[0, "text"]
    tr, te = split_annotations(labeled_reviews, seed=42)
    assert not set(labeled_reviews.iloc[tr]["text"]) & set(labeled_reviews.iloc[te]["text"])


def test_fasttext_retains_oov_subword_vectors():
    config = Config(embedding_dim=8, embedding_epochs=1, subword_buckets=1000)
    embedding = fit_embedding([["wisata", "pantai", "indah"], ["wisata", "nyaman", "bersih"]], config)
    assert "wisatanyaaaa" not in embedding.wv.key_to_index
    assert np.linalg.norm(embedding.wv["wisatanyaaaa"]) > 0


def test_negation_survives_preprocessing():
    assert "tidak" in TextPreprocessor()("Pantai ini tidak bersih")


def test_incomplete_labels_block_training(labeled_reviews, tmp_path):
    labeled_reviews.loc[0, "sentiment_label"] = ""
    with pytest.raises(ValueError, match="belum lengkap"):
        run_experiment(labeled_reviews, tmp_path / "model", Config())


def test_cli_rejects_unlabeled_workbook_template(labeled_reviews, tmp_path):
    labeled_reviews["annotation_status"] = "draft"
    path = tmp_path / "draft.csv"
    labeled_reviews.to_csv(path, index=False)
    run = subprocess.run([sys.executable, "-m", "wisata.train", str(path),
                          "--output", str(tmp_path / "model")], capture_output=True, text=True)
    assert run.returncode == 1
    assert "approved" in run.stderr
