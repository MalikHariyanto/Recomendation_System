"""Validasi dan persiapan data bersama, mengikuti notebook 1–3.

Modul ini tidak melatih model, menebak sentimen, atau membuat label musim.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO

import pandas as pd
from ftfy import fix_text


REQUIRED_COLUMNS = (
    "id_destinasi", "destinasi", "rating", "date", "text_original",
    "text_preprocessed", "rekomendasi_cuaca", "wisata",
)
SEASONS = {"Kemarau", "Penghujan", "Kemarau & Penghujan"}


class DatasetError(ValueError):
    """Data tidak dapat dipakai tanpa koreksi eksplisit dari pemilik data."""


@dataclass(frozen=True)
class Audit:
    original_rows: int
    empty_reviews: int
    short_reviews_removed: int
    duplicate_reviews_removed: int
    clean_reviews: int
    original_destinations: int
    clean_destinations: int
    categories: int


@dataclass
class PreparedData:
    reviews: pd.DataFrame
    catalog: pd.DataFrame
    audit: Audit


def read_excel(content: bytes) -> pd.DataFrame:
    """Baca sheet pertama tanpa mengubah file sumber atau menjalankan makro."""
    if len(content) > 25 * 1024 * 1024:
        raise DatasetError("Ukuran dataset melebihi 25 MB.")
    try:
        return pd.read_excel(BytesIO(content), engine="openpyxl")
    except Exception as exc:
        raise DatasetError("File harus berupa workbook .xlsx yang dapat dibaca.") from exc


def prepare_data(raw: pd.DataFrame) -> PreparedData:
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise DatasetError("Kolom wajib belum tersedia: " + ", ".join(missing))
    if raw.empty:
        raise DatasetError("Dataset tidak memiliki baris ulasan.")

    df = raw.loc[:, REQUIRED_COLUMNS].copy()
    for column in ("id_destinasi", "destinasi", "wisata", "rekomendasi_cuaca"):
        if df[column].isna().any():
            raise DatasetError(f"Kolom {column} memiliki nilai kosong.")
        df[column] = df[column].astype(str).str.strip()
        if df[column].eq("").any():
            raise DatasetError(f"Kolom {column} memiliki nilai kosong.")
    df["wisata"] = df["wisata"].str.lower()

    if not df["rekomendasi_cuaca"].isin(SEASONS).all():
        raise DatasetError("Label musim harus Kemarau, Penghujan, atau Kemarau & Penghujan.")
    for key, value in (("id_destinasi", "destinasi"), ("destinasi", "id_destinasi"),
                       ("destinasi", "wisata"), ("destinasi", "rekomendasi_cuaca")):
        if df.groupby(key)[value].nunique().gt(1).any():
            raise DatasetError(f"Pemetaan {key} ke {value} tidak konsisten.")

    ratings = pd.to_numeric(df["rating"], errors="coerce")
    if not ratings.isin([1, 2, 3, 4, 5]).all():
        raise DatasetError("Rating harus berupa bilangan bulat 1–5 tanpa nilai kosong.")
    df["rating"] = ratings.astype(int)
    df["text"] = (df["text_original"].fillna("").astype(str).map(fix_text)
                  .str.replace(r"\s+", " ", regex=True).str.strip())

    empty = int(df["text"].eq("").sum())
    short = df["text"].str.split().str.len().lt(3)
    original_destinations = df["destinasi"].nunique()
    df = df.loc[~short].copy()
    before_dedup = len(df)
    df = df.drop_duplicates(["destinasi", "text"])
    if df.empty:
        raise DatasetError("Tidak ada ulasan dengan sekurang-kurangnya tiga kata setelah pembersihan.")

    catalog = (df.groupby("destinasi", sort=True)
               .agg(id_destinasi=("id_destinasi", "first"),
                    wisata=("wisata", "first"), musim=("rekomendasi_cuaca", "first"),
                    n_ulasan=("text", "size"), rating_mean=("rating", "mean"))
               .reset_index())
    catalog["cocok_hujan"] = catalog["musim"].str.contains("Penghujan", regex=False)
    catalog["cocok_kemarau"] = catalog["musim"].str.contains("Kemarau", regex=False)
    order = dict(zip(catalog["destinasi"], range(len(catalog))))
    df["did"] = df["destinasi"].map(order)
    df = df.sort_values("did", kind="stable").reset_index(drop=True)
    audit = Audit(len(raw), empty, int(short.sum()), before_dedup - len(df), len(df),
                  original_destinations, len(catalog), catalog["wisata"].nunique())
    return PreparedData(df, catalog, audit)


def audit_payload(content: bytes, prepared: PreparedData) -> dict:
    return {
        "dataset_sha256": sha256(content).hexdigest(),
        "preprocessing": "original-text-ftfy-whitespace-min3-dedup-destination-text-v1",
        **asdict(prepared.audit),
        "destinations_by_season": {
            str(k): int(v) for k, v in prepared.catalog["musim"].value_counts().items()
        },
        "destinations_per_category": {
            str(k): int(v) for k, v in prepared.catalog["wisata"].value_counts().items()
        },
    }
