"""CLI: python -m wisata.audit /path/to/koreksi_dataset.xlsx --output report.json."""

import argparse
import json
from pathlib import Path

from wisata.data import DatasetError, audit_payload, prepare_data, read_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset ulasan wisata tanpa melatih model.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        content = args.dataset.read_bytes()
        result = audit_payload(content, prepare_data(read_excel(content)))
    except (DatasetError, OSError) as exc:
        parser.exit(1, f"Audit gagal: {exc}\n")
    report = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
