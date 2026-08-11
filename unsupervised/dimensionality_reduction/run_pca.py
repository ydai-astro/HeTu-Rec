#!/usr/bin/env python3
"""Reduce a feature TSV with PCA and save explained-variance diagnostics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from unsupervised.feature_io import read_feature_tsv, write_feature_tsv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--components", required=True, type=int)
    parser.add_argument("--metrics-output", default="outputs/metrics/pca_explained_variance.csv", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    names, matrix = read_feature_tsv(args.input)
    if args.components > min(matrix.shape):
        raise ValueError(f"--components={args.components} exceeds min(samples, features)={min(matrix.shape)}")
    pca = PCA(n_components=args.components, random_state=args.seed)
    embedding = pca.fit_transform(matrix)
    write_feature_tsv(args.output, names, embedding)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("component", "explained_variance_ratio", "cumulative_explained_variance_ratio"))
        writer.writeheader()
        cumulative = 0.0
        for index, ratio in enumerate(pca.explained_variance_ratio_, start=1):
            cumulative += float(ratio)
            writer.writerow({"component": index, "explained_variance_ratio": float(ratio), "cumulative_explained_variance_ratio": cumulative})
    print(f"Saved PCA embedding: {args.output} shape={embedding.shape}", flush=True)
    print(f"Explained variance: {cumulative:.6f}", flush=True)


if __name__ == "__main__":
    main()
