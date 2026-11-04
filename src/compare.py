"""Collect every results/<model>/metrics.json into one comparison table (results/RESULTS.md).

Usage:
    python -m src.compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import RESULTS

HEADER = ("| Model | Params (M) | Accuracy | Macro F1 | Tumor recall | Missed tumors | ECE |\n"
          "|---|---:|---:|---:|---:|---:|---:|")


def build_table(results_dir: Path) -> str:
    rows = []
    for f in sorted(results_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        rows.append(m)
    if not rows:
        raise SystemExit(f"No metrics found in {results_dir}/<model>/metrics.json. Train a model first.")
    rows.sort(key=lambda m: (-m["tumor_recall"], -m["macro_f1"]))
    lines = [HEADER]
    for m in rows:
        lines.append(
            f"| {m['model']} | {m['params_m']:.1f} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} | "
            f"{m['tumor_recall']:.1%} | {m['missed_tumors']} | {m['ece']:.3f} |"
        )
    n = rows[0]["n_test_images"]
    lines.append(f"\nTest set: {n} images after duplicate removal. Sorted by tumor recall, then macro F1. "
                 "ECE = expected calibration error (lower is better).")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=RESULTS)
    args = p.parse_args(argv)
    table = build_table(args.results)
    (args.results / "RESULTS.md").write_text("# Results\n\n" + table + "\n")
    print(table)
    return table


if __name__ == "__main__":
    main()
