import numpy as np
import pandas as pd
import pytest

from wisata.data import prepare_data
from wisata.labeling import derive_labels
from wisata.retrieval import ReviewIndex, build_index, evaluate_known_items


class FixtureEncoder:
    model_id = "fixture-only-not-a-research-model"
    revision = "1"
    max_length = 256

    def encode(self, texts):
        return np.array([[1., 0.] if "pantai" in text.lower() else [0., 1.] for text in texts])


def create_index(raw_reviews, tmp_path):
    scope = pd.DataFrame({"id_destinasi": [1,2], "destinasi": ["Pantai Contoh","Museum Contoh"],
                          "regency": ["Bangkalan","Sumenep"],"status":["included","included"],
                          "basis":["fixture","fixture"]})
    labels = derive_labels(prepare_data(raw_reviews).reviews, scope)
    encoder = FixtureEncoder()
    build_index(labels, tmp_path / "index", encoder)
    return ReviewIndex(tmp_path / "index", encoder=encoder)


def test_cosine_ranking_and_season_exclude_invalid_candidates(raw_reviews, tmp_path):
    index = create_index(raw_reviews, tmp_path)
    assert index.search("pantai", k=10)["destinasi"].tolist()[0] == "Pantai Contoh"
    rainy = index.search("pantai", season="Hujan", k=10)
    assert rainy["destinasi"].tolist() == ["Museum Contoh"]
    assert index.search("pantai", season="Hujan", category="pantai").empty
    assert index.search("pantai", sentiment="negatif").empty


def test_blank_query_rejected(raw_reviews, tmp_path):
    with pytest.raises(ValueError, match="Query"):
        create_index(raw_reviews, tmp_path).search(" ")


def test_checksum_detects_wrong_metadata(raw_reviews, tmp_path):
    create_index(raw_reviews, tmp_path)
    (tmp_path / "index" / "reviews.json").write_text("[]")
    with pytest.raises(ValueError, match="Checksum"):
        ReviewIndex(tmp_path / "index", encoder=FixtureEncoder())


def test_leave_query_out_never_recommends_a_single_review_destination(raw_reviews, tmp_path):
    index = create_index(raw_reviews, tmp_path)
    row = index.reviews.loc[index.reviews["destinasi"].eq("Pantai Contoh")].iloc[0]
    result = index.search_vector(np.array([1., 0.]), exclude_review_id=row["review_id"])
    assert result["destinasi"].tolist() == ["Museum Contoh"]


def test_known_item_metrics_use_destination_ground_truth_and_exclude_query(raw_reviews, tmp_path):
    extra = raw_reviews.iloc[[0, 4]].copy()
    extra["text_original"] = ["Pantai indah untuk keluarga", "Museum bersejarah sangat menarik"]
    index = create_index(pd.concat([raw_reviews, extra], ignore_index=True), tmp_path)
    report = evaluate_known_items(index, k=2)
    assert report["queries"] == 2
    assert report["precision_at_k"] == .5
    assert report["recall_at_k"] == 1.
    assert report["mrr_at_k"] == 1.
