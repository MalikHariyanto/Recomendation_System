from io import BytesIO
import json
import subprocess
import sys

import pandas as pd
import pytest

from wisata.data import DatasetError, audit_payload, prepare_data, read_excel


def workbook(df):
    out = BytesIO()
    df.to_excel(out, index=False)
    return out.getvalue()


def test_cleanup_and_ratings_are_computed_after_filtering(raw_reviews):
    original = raw_reviews.copy(deep=True)
    result = prepare_data(raw_reviews)
    assert result.audit.original_rows == 5
    assert result.audit.empty_reviews == 1
    assert result.audit.short_reviews_removed == 2
    assert result.audit.duplicate_reviews_removed == 1
    assert result.audit.clean_reviews == 2
    assert result.catalog.set_index("destinasi").loc["Pantai Contoh", "rating_mean"] == 5
    pd.testing.assert_frame_equal(original, raw_reviews)


def test_dedup_preserves_same_text_at_different_destinations(raw_reviews):
    raw_reviews.loc[4, "text_original"] = raw_reviews.loc[0, "text_original"]
    assert prepare_data(raw_reviews).audit.clean_reviews == 2


def test_season_is_label_metadata_not_weather_forecast(raw_reviews):
    catalog = prepare_data(raw_reviews).catalog.set_index("destinasi")
    assert bool(catalog.loc["Pantai Contoh", "cocok_kemarau"])
    assert not bool(catalog.loc["Pantai Contoh", "cocok_hujan"])
    assert bool(catalog.loc["Museum Contoh", "cocok_hujan"])


def test_reviews_are_sorted_into_catalog_segments(raw_reviews):
    result = prepare_data(raw_reviews.sample(frac=1, random_state=1))
    assert result.reviews["did"].tolist() == [0, 1]
    assert result.reviews.groupby("did")["destinasi"].first().tolist() == result.catalog["destinasi"].tolist()


@pytest.mark.parametrize("rating", [None, "buruk", 0, 6, 4.5, float("inf")])
def test_invalid_rating_is_not_silently_imputed(raw_reviews, rating):
    raw_reviews["rating"] = raw_reviews["rating"].astype(object)
    raw_reviews.loc[0, "rating"] = rating
    with pytest.raises(DatasetError, match="Rating"):
        prepare_data(raw_reviews)


@pytest.mark.parametrize("column,value", [
    ("destinasi", "Nama berbeda"), ("id_destinasi", 999),
    ("wisata", "religi"), ("rekomendasi_cuaca", "Penghujan"),
])
def test_conflicting_metadata_blocks_analysis(raw_reviews, column, value):
    raw_reviews.loc[1, column] = value
    with pytest.raises(DatasetError, match="tidak konsisten"):
        prepare_data(raw_reviews)


@pytest.mark.parametrize("column", ["destinasi", "id_destinasi", "wisata", "rekomendasi_cuaca"])
def test_empty_identity_and_metadata_are_rejected(raw_reviews, column):
    raw_reviews[column] = raw_reviews[column].astype(object)
    raw_reviews.loc[0, column] = "  "
    with pytest.raises(DatasetError, match="kosong"):
        prepare_data(raw_reviews)


def test_missing_column_and_empty_workbook(raw_reviews):
    with pytest.raises(DatasetError, match="Kolom wajib"):
        prepare_data(raw_reviews.drop(columns="text_original"))
    with pytest.raises(DatasetError, match="tidak memiliki"):
        prepare_data(raw_reviews.iloc[:0])


def test_all_reviews_filtered_out(raw_reviews):
    raw_reviews["text_original"] = "ok"
    with pytest.raises(DatasetError, match="Tidak ada ulasan"):
        prepare_data(raw_reviews)


def test_unknown_season_is_not_inferred(raw_reviews):
    raw_reviews.loc[0, "rekomendasi_cuaca"] = "libur sekolah"
    with pytest.raises(DatasetError, match="Label musim"):
        prepare_data(raw_reviews)


def test_invalid_excel_has_actionable_error():
    with pytest.raises(DatasetError, match="workbook"):
        read_excel(b"not an excel file")


def test_audit_workbook_and_cli_roundtrip(raw_reviews, tmp_path):
    content = workbook(raw_reviews)
    prepared = prepare_data(read_excel(content))
    payload = audit_payload(content, prepared)
    assert payload["clean_reviews"] == 2
    assert payload["categories"] == 2
    assert payload["destinations_per_category"] == {"pantai": 1, "sejarah": 1}
    assert len(payload["dataset_sha256"]) == 64
    assert payload["destinations_by_season"] == {"Kemarau": 1, "Kemarau & Penghujan": 1}
    path = tmp_path / "reviews.xlsx"
    path.write_bytes(content)
    dest = tmp_path / "report.json"
    run = subprocess.run([sys.executable, "-m", "wisata.audit", str(path), "--output", str(dest)],
                         capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == json.loads(dest.read_text()) == payload


def test_cli_failure_preserves_nonzero_status(tmp_path):
    run = subprocess.run([sys.executable, "-m", "wisata.audit", str(tmp_path / "missing.xlsx")],
                         capture_output=True, text=True)
    assert run.returncode == 1
    assert "Audit gagal" in run.stderr
