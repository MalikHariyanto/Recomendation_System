import pandas as pd
import pytest


@pytest.fixture
def raw_reviews():
    return pd.DataFrame([
        [1, "Pantai Contoh", 5, "2026-01-01", "Pantai bersih dan tenang", "pantai bersih", "Kemarau", "pantai"],
        [1, "Pantai Contoh", 4, "2026-01-02", "Pantai  bersih dan tenang", "pantai bersih", "Kemarau", "pantai"],
        [1, "Pantai Contoh", 3, "2026-01-03", "bagus", "bagus", "Kemarau", "pantai"],
        [1, "Pantai Contoh", 2, "2026-01-04", None, None, "Kemarau", "pantai"],
        [2, "Museum Contoh", 4, "2026-01-01", "Museum nyaman untuk keluarga", "museum nyaman", "Kemarau & Penghujan", "sejarah"],
    ], columns=["id_destinasi", "destinasi", "rating", "date", "text_original",
                "text_preprocessed", "rekomendasi_cuaca", "wisata"])
