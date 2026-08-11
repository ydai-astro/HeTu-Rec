#!/usr/bin/env python3
"""Cluster a feature TSV and write assignments plus optional image folders."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering, Birch, KMeans

from unsupervised.feature_io import read_feature_tsv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", default=("kmeans", "agglomerative", "birch"), choices=("kmeans", "agglomerative", "birch"))
    parser.add_argument("--clusters", required=True, type=int)
    parser.add_argument("--image-root", type=Path, help="Optional root used to copy images into cluster folders.")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def build_model(method: str, clusters: int, seed: int):
    if method == "kmeans":
        return KMeans(n_clusters=clusters, random_state=seed, n_init=10)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=clusters, linkage="ward")
    return Birch(n_clusters=clusters)


def main():
    args = parse_args()
    names, matrix = read_feature_tsv(args.input)
    for method in args.methods:
        labels = build_model(method, args.clusters, args.seed).fit_predict(matrix)
        method_dir = args.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        assignment_path = method_dir / "assignments.csv"
        with assignment_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("relative_path", "cluster"))
            writer.writeheader()
            writer.writerows({"relative_path": name, "cluster": int(label)} for name, label in zip(names, labels))

        copied = 0
        if args.image_root:
            for name, label in zip(names, labels):
                source = args.image_root / name
                if not source.is_file():
                    raise FileNotFoundError(f"Feature path is absent below --image-root: {source}")
                destination = method_dir / "clusters" / str(label) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied += 1
        print(f"method={method} assignments={assignment_path} samples={len(names)} copied_images={copied}", flush=True)


if __name__ == "__main__":
    main()
