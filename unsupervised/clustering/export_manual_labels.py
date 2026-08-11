#!/usr/bin/env python3
"""Convert manually reviewed consensus clusters into filename-prefixed labels."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--cluster-labels", required=True, type=Path, help="JSON mapping from cluster ID to final integer class.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = {str(key): int(value) for key, value in json.loads(args.cluster_labels.read_text(encoding="utf-8")).items()}
    with args.assignments.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    written, skipped = 0, 0
    for row in rows:
        cluster = row["consensus_cluster"]
        if cluster not in labels:
            skipped += 1
            continue
        source = args.image_root / row["relative_path"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing image: {source}")
        destination = args.output_dir / f"{labels[cluster]}_{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        written += 1
    print(f"written={written} skipped_unmapped_clusters={skipped} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
