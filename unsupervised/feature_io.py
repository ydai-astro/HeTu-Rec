from __future__ import annotations

from pathlib import Path

import numpy as np


def read_feature_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    names, vectors = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"Expected two tab-separated fields at line {line_number}: {path}")
            names.append(fields[0])
            vectors.append([float(value) for value in fields[1].split("<=>")])
    if not vectors:
        raise RuntimeError(f"No features found in: {path}")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid feature matrix in: {path}")
    return names, matrix


def write_feature_tsv(path: Path, names: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, vector in zip(names, vectors):
            values = "<=>".join(str(float(value)) for value in vector)
            handle.write(f"{name}\t{values}\n")
