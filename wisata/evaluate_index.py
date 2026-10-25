"""Evaluasi known-item terbatas; tidak memerlukan encoding ulang atau human qrels."""

import argparse
import json
from pathlib import Path

from wisata.retrieval import ReviewIndex, evaluate_known_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    result = evaluate_known_items(ReviewIndex(args.index), k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
