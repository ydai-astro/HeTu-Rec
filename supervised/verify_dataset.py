#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    data_root = Path("Data")
    summary_path = data_root / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Missing Data/dataset_summary.json. Run prepare_dataset.py first.")
    image_size = tuple(json.loads(summary_path.read_text(encoding="utf-8"))["image_size"])
    report = {}
    for split in ("train", "val", "test"):
        rows = read_csv(data_root / "splits" / f"{split}.csv")
        labels = Counter(int(row["label"]) for row in rows)
        missing = []
        invalid = []
        for row in rows:
            path = data_root / row["relative_path"]
            if not path.is_file():
                missing.append(str(path))
                continue
            with Image.open(path) as image:
                if image.size != image_size:
                    invalid.append(str(path))
            expected_label = int(Path(row["relative_path"]).name.split("_", 1)[0])
            if expected_label != int(row["label"]):
                invalid.append(str(path))
        report[split] = {
            "samples": len(rows),
            "labels": dict(sorted(labels.items())),
            "missing": len(missing),
            "invalid": len(invalid),
        }
        if missing or invalid:
            raise RuntimeError(f"Dataset verification failed for {split}.")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
