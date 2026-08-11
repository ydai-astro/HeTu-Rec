#!/usr/bin/env python3
"""Search UMAP dimensions by Davies-Bouldin score after K-means clustering."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from unsupervised.feature_io import read_feature_tsv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default="outputs/metrics/umap_db_scores.csv", type=Path)
    parser.add_argument("--dimensions", nargs="+", required=True, type=int)
    parser.add_argument("--clusters", required=True, type=int)
    parser.add_argument("--neighbors", default=100, type=int)
    parser.add_argument("--min-dist", default=0.1, type=float)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    _, matrix = read_feature_tsv(args.input)
    rows = []
    for dimensions in args.dimensions:
        reducer = umap.UMAP(n_components=dimensions, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric, random_state=args.seed)
        embedding = reducer.fit_transform(matrix)
        labels = KMeans(n_clusters=args.clusters, random_state=args.seed, n_init=10).fit_predict(embedding)
        score = davies_bouldin_score(embedding, labels)
        rows.append({"components": dimensions, "davies_bouldin": score})
        print(f"components={dimensions} db={score:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("components", "davies_bouldin"))
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda row: row["davies_bouldin"])
    print(f"Best components={best['components']} db={best['davies_bouldin']:.6f}", flush=True)


if __name__ == "__main__":
    main()
