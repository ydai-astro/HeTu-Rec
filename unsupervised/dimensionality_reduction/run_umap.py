#!/usr/bin/env python3
"""Reduce a feature TSV with UMAP using fully relative paths."""
from __future__ import annotations

import argparse
from pathlib import Path

import umap

from unsupervised.feature_io import read_feature_tsv, write_feature_tsv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--components", required=True, type=int)
    parser.add_argument("--neighbors", default=100, type=int)
    parser.add_argument("--min-dist", default=0.1, type=float)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    names, matrix = read_feature_tsv(args.input)
    reducer = umap.UMAP(n_components=args.components, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric, random_state=args.seed)
    embedding = reducer.fit_transform(matrix)
    write_feature_tsv(args.output, names, embedding)
    print(f"Saved UMAP embedding: {args.output} shape={embedding.shape}", flush=True)


if __name__ == "__main__":
    main()
