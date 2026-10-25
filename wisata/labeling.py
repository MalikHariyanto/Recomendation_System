"""Label turunan sesuai keputusan pengguna; tidak menyatakan anotasi manusia."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from wisata.annotations import annotation_template, validate_annotations
from wisata.data import DatasetError, prepare_data, read_excel


DEFAULT_SCOPE = Path(__file__).resolve().parents[1] / "config" / "madura_destinations.csv"
POLICY = {
    "version": "dataset-labels-v1",
    "sentiment": "rating 1-2 negatif; 3 netral; 4-5 positif",
    "season": "rekomendasi_cuaca; kelas asli, bukan tanggal kunjungan terverifikasi",
    "tourism": "wisata; kategori asli dataset tanpa penggabungan otomatis",
    "region": "Madura sesuai proposal kolaborasi; manifest identitas destinasi",
}


def scoped_reviews(reviews: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    required = {"id_destinasi", "destinasi", "regency", "status", "basis"}
    if not required.issubset(scope.columns):
        raise DatasetError("Manifest wilayah belum memiliki semua kolom wajib.")
    manifest = scope.copy()
    manifest["key"] = manifest["id_destinasi"].astype(str).str.replace(r"\.0$", "", regex=True)
    if manifest["key"].duplicated().any():
        raise DatasetError("ID dalam manifest wilayah harus unik.")
    if not manifest["status"].isin(["included", "pending_location"]).all():
        raise DatasetError("Status wilayah tidak dikenal.")
    if not manifest["regency"].isin(["Bangkalan", "Sampang", "Pamekasan", "Sumenep"]).all():
        raise DatasetError("Kabupaten manifest harus berada dalam cakupan Madura.")
    df = reviews.copy()
    df["key"] = df["id_destinasi"].astype(str).str.replace(r"\.0$", "", regex=True)
    merged = df.merge(manifest[["key", "destinasi", "regency", "status", "basis"]], on="key",
                      how="inner", suffixes=("", "_scope"), validate="many_to_one")
    if not merged["destinasi"].eq(merged["destinasi_scope"]).all():
        raise DatasetError("Nama destinasi tidak cocok dengan ID manifest wilayah; periksa versi dataset.")
    return merged.loc[merged["status"].eq("included")].drop(columns=["key", "destinasi_scope"]).reset_index(drop=True)


def derive_labels(reviews: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    subset = scoped_reviews(reviews, scope)
    if subset.empty:
        raise DatasetError("Tidak ada destinasi yang cocok dengan manifest wilayah Madura.")
    labels = annotation_template(subset)
    labels["season_label"] = subset["rekomendasi_cuaca"].to_numpy()
    labels["tourism_label"] = subset["wisata"].to_numpy()
    labels["sentiment_label"] = subset["rating"].map({1: "negatif", 2: "negatif", 3: "netral",
                                                     4: "positif", 5: "positif"}).to_numpy()
    labels["region"] = "madura"
    labels["regency"] = subset["regency"].to_numpy()
    labels["id_destinasi"] = subset["id_destinasi"].to_numpy()
    labels["annotation_status"] = "derived"
    labels["sentiment_source"] = "rating"
    labels["season_source"] = "rekomendasi_cuaca"
    labels["tourism_source"] = "wisata"
    labels["label_policy"] = POLICY["version"]
    labels["notes"] = "Label turunan metadata sesuai keputusan pengguna; bukan anotasi teks manusia."
    return validate_annotations(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Siapkan label rating/cuaca/kategori dalam cakupan Madura.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.dataset.read_bytes()
    prepared = prepare_data(read_excel(content))
    scope_content = args.scope.read_bytes()
    labels = derive_labels(prepared.reviews, pd.read_csv(args.scope))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)
    report = {"policy": POLICY, "dataset_sha256": sha256(content).hexdigest(),
              "scope_sha256": sha256(scope_content).hexdigest(), "rows": len(labels),
              "destinations": labels["destinasi"].nunique(),
              "regencies": labels.groupby("regency")["destinasi"].nunique().to_dict(),
              "label_counts": {column: labels[column].value_counts().to_dict()
                               for column in ["season_label", "tourism_label", "sentiment_label"]}}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
