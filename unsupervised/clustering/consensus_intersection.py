#!/usr/bin/env python3
"""Keep samples whose cluster assignment agrees after majority label alignment."""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def read_assignments(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty assignment file: {path}")
    return {row["relative_path"]: row["cluster"] for row in rows}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--others", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    anchor = read_assignments(args.anchor)
    others = [read_assignments(path) for path in args.others]
    shared = set(anchor)
    for assignments in others:
        shared &= set(assignments)
    if not shared:
        raise RuntimeError("No samples are shared across all assignment files.")

    aligned = []
    for assignments in others:
        mapping_counts = defaultdict(Counter)
        for name in shared:
            mapping_counts[assignments[name]][anchor[name]] += 1
        mapping = {cluster: counts.most_common(1)[0][0] for cluster, counts in mapping_counts.items()}
        aligned.append({name: mapping[assignments[name]] for name in shared})

    retained = [name for name in sorted(shared) if all(labels[name] == anchor[name] for labels in aligned)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "consensus_cluster"))
        writer.writeheader()
        writer.writerows({"relative_path": name, "consensus_cluster": anchor[name]} for name in retained)

    copied = 0
    if args.image_root:
        for name in retained:
            source = args.image_root / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing image: {source}")
            destination = args.output.parent / "clusters" / anchor[name] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    print(f"shared={len(shared)} retained_consensus={len(retained)} copied_images={copied}", flush=True)


if __name__ == "__main__":
    main()
