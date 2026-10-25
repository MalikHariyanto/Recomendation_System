import pandas as pd
import pytest

from wisata.annotations import validate_annotations
from wisata.data import DatasetError, prepare_data
from wisata.labeling import derive_labels, scoped_reviews


def manifest():
    return pd.DataFrame({"id_destinasi": [1, 2], "destinasi": ["Pantai Contoh", "Museum Contoh"],
                         "regency": ["Bangkalan", "Sumenep"], "status": ["included", "included"],
                         "basis": ["synthetic fixture", "synthetic fixture"]})


def test_labels_follow_user_policy_and_keep_provenance(raw_reviews):
    reviews = prepare_data(raw_reviews).reviews
    result = derive_labels(reviews, manifest())
    assert set(result["season_label"]) == {"Kemarau", "Kemarau & Penghujan"}
    assert set(result["tourism_label"]) == {"pantai", "sejarah"}
    assert result["annotation_status"].eq("derived").all()
    assert result["sentiment_source"].eq("rating").all()
    assert result["region"].eq("madura").all()


@pytest.mark.parametrize("rating,expected", [(1,"negatif"),(2,"negatif"),(3,"netral"),(4,"positif"),(5,"positif")])
def test_rating_sentiment_boundaries(raw_reviews, rating, expected):
    reviews = prepare_data(raw_reviews).reviews
    reviews["rating"] = rating
    assert derive_labels(reviews, manifest())["sentiment_label"].eq(expected).all()


def test_pending_locations_are_excluded_without_changing_original(raw_reviews):
    scope = manifest()
    scope.loc[0, "status"] = "pending_location"
    reviews = prepare_data(raw_reviews).reviews
    assert scoped_reviews(reviews, scope)["destinasi"].tolist() == ["Museum Contoh"]
    assert len(reviews) == 2


def test_manifest_identity_mismatch_is_not_silently_accepted(raw_reviews):
    scope = manifest()
    scope.loc[0,"destinasi"] = "Nama lain"
    with pytest.raises(DatasetError, match="Nama destinasi"):
        scoped_reviews(prepare_data(raw_reviews).reviews, scope)


def test_derived_status_cannot_impersonate_human_annotations(raw_reviews):
    result = derive_labels(prepare_data(raw_reviews).reviews, manifest()).drop(columns="sentiment_source")
    with pytest.raises(ValueError, match="Provenance"):
        validate_annotations(result)
