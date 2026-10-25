"""Jalankan tiga skenario identik splitnya dan simpan ringkasan hasil nyata."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import pandas as pd

from wisata.annotations import LABELS, validate_annotations
from wisata.training import Config, TextPreprocessor, run_experiment, token_cache_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    df = validate_annotations(pd.read_csv(args.labels)).reset_index(drop=True)
    if not df["region"].eq("madura").all():
        parser.error("Runner penelitian utama memerlukan label dalam cakupan Madura.")
    args.output.mkdir(parents=True, exist_ok=True)
    key = token_cache_key(df["text"].tolist())
    cache = args.output / f"tokens-{key}.json"
    if cache.exists():
        tokens = json.loads(cache.read_text())
    else:
        processor = TextPreprocessor()
        tokens = []
        for i, text in enumerate(df["text"]):
            tokens.append(processor(text))
            if i % 500 == 0:
                print(f"Preprocessing {i}/{len(df)}", flush=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, cache)
    records = []
    reference = None
    for scenario in ["fasttext", "lda", "fusion"]:
        directory = args.output / scenario
        manifest = directory / "manifest.json"
        config = Config(scenario=scenario, epochs=args.epochs)
        if manifest.exists():
            from dataclasses import asdict
            result = json.loads(manifest.read_text())
            if (result["annotation_sha256"] != sha256(df.to_csv(index=False).encode()).hexdigest()
                    or result["config"] != asdict(config)):
                raise ValueError("Hasil lama tidak cocok dengan label/konfigurasi saat ini; gunakan folder baru.")
        else:
            result = run_experiment(df, directory, config, preprocessed_tokens=tokens)
        split = (result["train_review_ids"], result["test_review_ids"])
        if reference is not None and split != reference:
            raise RuntimeError("Split tiga skenario harus identik.")
        reference = split
        records.append({"scenario": scenario, "train_rows": result["train_rows"],
                        "test_rows": result["test_rows"],
                        "overall_hamming_loss": result["metrics"]["overall_hamming_loss"],
                        **{label + "_" + metric: result["metrics"][label][metric]
                           for label in LABELS for metric in ["micro_f1", "macro_f1", "hamming_loss"]}})
        pd.DataFrame(records).to_csv(args.output / "summary.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
