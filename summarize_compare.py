#!/usr/bin/env python3
"""
Quickly turn compare.json into a Markdown table for README/reports.

Usage:
  python summarize_compare.py results/exp_20251201T104924Z/compare.json
"""
import json
import sys
from pathlib import Path


def load_metrics(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = [data]
    return data


def to_markdown_rows(metrics):
    rows = []
    for m in metrics:
        rows.append(
            "| {preset:<12} | {acc:.4f} | {nbo:.2f}% | {top2:.4f} | {top3:.4f} | {aug} |".format(
                preset=m.get("preset", m.get("model", "?")),
                acc=m["test_accuracy"],
                nbo=m["next_best_option"]["overall_percentage"],
                top2=m["top2_accuracy"],
                top3=m["top3_accuracy"],
                aug=m.get("augmentation", "n/a"),
            )
        )
    return rows


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip())
        sys.exit(1)

    path = Path(sys.argv[1])
    metrics = load_metrics(path)

    header = "| Model        | Test acc | Next-Best Option | Top-2 | Top-3 | Augmentation |"
    sep = "|--------------|----------|------------------|-------|-------|--------------|"
    rows = to_markdown_rows(metrics)

    print(header)
    print(sep)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
