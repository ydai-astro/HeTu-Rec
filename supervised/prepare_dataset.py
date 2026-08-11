#!/usr/bin/env python3
"""Create a reproducible 8:1:1 binary split from numbered PNG folders."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--positive-folders", nargs="+", required=True, type=int)
    parser.add_argument("--num-folders", default=50, type=int)
    parser.add_argument("--image-size", default=100, type=int)
    parser.add_argument("--rest-folder", default="rest")
    parser.add_argument("--skip-rest", action="store_true")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def png_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.png") if path.is_file())


def validate_png(path: Path, image_size: int) -> None:
    with Image.open(path) as image:
        image.load()
        if image.size != (image_size, image_size):
            raise ValueError(f"Expected {image_size}x{image_size} PNG: {path}")


def split_rows(rows: list[dict[str, object]], seed: int) -> None:
    rng = random.Random(seed)
    rng.shuffle(rows)
    validation_count = round(len(rows) * 0.1)
    test_count = round(len(rows) * 0.1)
    train_count = len(rows) - validation_count - test_count
    for index, row in enumerate(rows):
        row["split"] = "train" if index < train_count else "val" if index < train_count + validation_count else "test"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    project_root = args.project_root.resolve()
    data_root = project_root / "Data"
    image_root = data_root / "images"
    rest_root = data_root / "rest"
    positive_folders = set(args.positive_folders)

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if any(data_root.rglob("*.png")):
        raise RuntimeError("Refusing to overwrite an existing prepared Data directory.")
    if not positive_folders.issubset(set(range(args.num_folders))):
        raise ValueError("Every positive folder must be within [0, num_folders).")

    rows: list[dict[str, object]] = []
    for folder_id in range(args.num_folders):
        folder = source / str(folder_id)
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing numbered folder: {folder}")
        label = int(folder_id in positive_folders)
        folder_rows: list[dict[str, object]] = []
        for path in png_paths(folder):
            validate_png(path, args.image_size)
            relative_to_folder = path.relative_to(folder)
            destination_name = f"{label}_{folder_id:02d}_{path.name}"
            relative_path = Path("images") / f"folder_{folder_id:02d}" / relative_to_folder.parent / destination_name
            folder_rows.append({
                "relative_path": relative_path.as_posix(),
                "label": label,
                "split": "",
                "source_folder": folder_id,
                "original_relative_path": relative_to_folder.as_posix(),
            })
        if not folder_rows:
            raise RuntimeError(f"No PNG files found in: {folder}")
        split_rows(folder_rows, args.seed + folder_id)
        rows.extend(folder_rows)

    for row in rows:
        source_path = source / str(row["source_folder"]) / str(row["original_relative_path"])
        destination = data_root / str(row["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    rest_rows: list[dict[str, str]] = []
    if not args.skip_rest:
        rest_source = source / args.rest_folder
        if not rest_source.is_dir():
            raise FileNotFoundError(f"Unlabeled folder not found: {rest_source}")
        for path in png_paths(rest_source):
            validate_png(path, args.image_size)
            relative_path = path.relative_to(rest_source)
            destination = rest_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            rest_rows.append({"relative_path": relative_path.as_posix()})

    fields = ["relative_path", "label", "split", "source_folder", "original_relative_path"]
    rows.sort(key=lambda row: (str(row["split"]), int(row["source_folder"]), str(row["relative_path"])))
    write_csv(data_root / "manifest.csv", rows, fields)
    for split in SPLITS:
        write_csv(data_root / "splits" / f"{split}.csv", [row for row in rows if row["split"] == split], fields)
    write_csv(data_root / "rest_manifest.csv", rest_rows, ["relative_path"])

    labels = Counter(int(row["label"]) for row in rows)
    summary = {
        "positive_source_folders": sorted(positive_folders),
        "seed": args.seed,
        "image_size": [args.image_size, args.image_size],
        "labeled_image_count": len(rows),
        "class_counts": {"0": labels[0], "1": labels[1]},
        "rest_image_count": len(rest_rows),
        "split_counts": {
            split: {
                "total": sum(row["split"] == split for row in rows),
                "class_0": sum(row["split"] == split and row["label"] == 0 for row in rows),
                "class_1": sum(row["split"] == split and row["label"] == 1 for row in rows),
            }
            for split in SPLITS
        },
        "split_rule": "Each numbered source folder is independently split 8:1:1.",
    }
    (data_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
