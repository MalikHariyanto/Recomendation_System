import pytest

from wisata.annotations import annotation_template, validate_annotations
from wisata.data import prepare_data


def test_template_preserves_label_sources_without_inventing_targets(raw_reviews):
    template = annotation_template(prepare_data(raw_reviews).reviews)
    assert len(template) == 2
    assert template["review_id"].nunique() == 2
    assert template["source_rating"].notna().all()
    assert template["source_weather_suitability"].notna().all()
    assert template["sentiment_label"].eq("").all()
    assert template["season_label"].eq("").all()
    with pytest.raises(ValueError, match="belum lengkap"):
        validate_annotations(template)
