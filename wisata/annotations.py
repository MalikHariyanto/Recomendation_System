"""Ekspor template anotasi; label target tidak pernah diturunkan dari rating."""

import argparse
from hashlib import sha256
from pathlib import Path

import pandas as pd

from wisata.data import prepare_data, read_excel


LABELS = ("season_label", "tourism_label", "sentiment_label")
SEASON_LABELS = {"Kemarau", "Penghujan", "Kemarau & Penghujan"}
SENTIMENT_LABELS = {"positif", "negatif", "netral"}
TOURISM_LABELS = {"sejarah", "religi", "lainnya", "pantai", "air terjun", "rekreasi/edukasi",
                  "pegunungan", "taman", "hutan/perkebunan", "goa", "desa/kampung wisata",
                  "danau/waduk", "sungai", "pemandian"}


def annotation_template(reviews: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    sample = reviews if limit is None else reviews.sample(min(limit, len(reviews)), random_state=42)
    out = pd.DataFrame({
        "review_id": [sha256(f"{row.id_destinasi}\0{row.text}".encode()).hexdigest()
                      for row in sample.itertuples()],
        "text": sample["text"].tolist(),
        "destinasi": sample["destinasi"].tolist(),
        "source_date": sample["date"].astype(str).tolist(),
        "source_rating": sample["rating"].tolist(),
        "source_weather_suitability": sample["rekomendasi_cuaca"].tolist(),
    })
    for column in (*LABELS, "region", "annotation_status", "annotator_code", "notes"):
        out[column] = ""
    return out


def validate_annotations(df: pd.DataFrame) -> pd.DataFrame:
    required = {"review_id", "text", "region", "annotation_status", *LABELS}
    if not required.issubset(df.columns):
        raise ValueError("Kolom anotasi belum lengkap: " + ", ".join(sorted(required - set(df.columns))))
    result = df.copy()
    for column in required:
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Anotasi {column} belum lengkap; jangan gunakan label buatan untuk menutup kekosongan.")
        result[column] = result[column].astype(str).str.strip()
    if result["review_id"].duplicated().any():
        raise ValueError("review_id harus unik.")
    for column, allowed in (("season_label", SEASON_LABELS), ("tourism_label", TOURISM_LABELS),
                            ("sentiment_label", SENTIMENT_LABELS)):
        if not result[column].isin(allowed).all():
            raise ValueError(f"Kategori {column} harus mengikuti pedoman anotasi.")
    if not result["annotation_status"].isin(["approved", "derived"]).all():
        raise ValueError("Label training harus berstatus approved atau derived dengan provenance.")
    derived = result["annotation_status"].eq("derived")
    if derived.any():
        for column, source in (("sentiment_source", "rating"), ("season_source", "rekomendasi_cuaca"),
                               ("tourism_source", "wisata")):
            if column not in result or not result.loc[derived, column].eq(source).all():
                raise ValueError("Provenance label derived tidak lengkap atau tidak sesuai kebijakan.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Siapkan anotasi kosong, tanpa pseudo-label.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit harus positif")
    data = prepare_data(read_excel(args.dataset.read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotation_template(data.reviews, args.limit).to_csv(args.output, index=False)
    print(f"Template disimpan di {args.output}; label masih kosong dan belum boleh dipakai training.")


if __name__ == "__main__":
    main()
