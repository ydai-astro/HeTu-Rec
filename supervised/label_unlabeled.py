#!/usr/bin/env python3
"""Classify unlabeled PNG files and copy them into class-specific folders."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from data import make_transform
from model import build_model


class UnlabeledPngDataset(Dataset):
    def __init__(self, input_dir: Path, paths: list[Path], transform):
        self.input_dir = input_dir
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.relative_to(self.input_dir).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict unlabeled PNG files with the selected DINOv3 binary classifier."
    )
    parser.add_argument("--input-dir", default="Data/rest", type=Path)
    parser.add_argument("--output-dir", default="Data/not_label", type=Path)
    parser.add_argument(
        "--checkpoint",
        default="runs/full_d4_seed42/best_checkpoint.pt",
        type=Path,
    )
    parser.add_argument("--dinov3-repo", default="../external/dinov3", type=Path)
    parser.add_argument(
        "--weights",
        default="../weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        type=Path,
    )
    parser.add_argument("--threshold", type=float, help="Optional fixed class-1 threshold.")
    parser.add_argument(
        "--threshold-file",
        type=Path,
        help="Validation-calibrated threshold JSON written by run_test.sh.",
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--disable-tta", action="store_true")
    return parser.parse_args()


def collect_pngs(input_dir: Path) -> list[Path]:
    paths = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not paths:
        raise RuntimeError(f"No PNG files found below: {input_dir.resolve()}")
    return paths


def assert_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir.resolve()}. "
            "Move or remove the prior prediction output before starting a new run."
        )


@torch.no_grad()
def predict_probabilities(
    model,
    input_dir: Path,
    paths: list[Path],
    image_size: int,
    batch_size: int,
    workers: int,
    device,
    amp_dtype,
    use_tta: bool,
) -> tuple[list[str], np.ndarray]:
    operations = range(8) if use_tta else [None]
    all_probabilities = None
    reference_paths = None

    for pass_index, operation in enumerate(operations, start=1):
        print(f"Inference pass {pass_index}/{len(operations)}", flush=True)
        dataset = UnlabeledPngDataset(
            input_dir,
            paths,
            make_transform(
                image_size=image_size,
                train=False,
                d4=False,
                tta_index=operation,
            ),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        probabilities = []
        relative_paths = []
        for images, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if device.type == "cuda" else torch.float32,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
            probabilities.extend(
                torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
            )
            relative_paths.extend(batch_paths)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if reference_paths is None:
            reference_paths = relative_paths
            all_probabilities = probabilities
        else:
            if reference_paths != relative_paths:
                raise RuntimeError("Input order changed between test-time augmentation passes.")
            all_probabilities += probabilities

    assert reference_paths is not None and all_probabilities is not None
    return reference_paths, all_probabilities / len(operations)


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("Output directory must not be inside the input directory.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint.resolve()}")
    assert_empty_output(output_dir)

    paths = collect_pngs(input_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    if args.threshold_file:
        threshold = float(json.loads(args.threshold_file.read_text(encoding="utf-8"))["threshold"])
    elif args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(checkpoint["validation_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("The selected threshold must be strictly between 0 and 1.")
    image_size = int(config["image_size"])
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = build_model(args.dinov3_repo, args.weights, dropout=float(config["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.to(device).eval()

    print(
        f"Unlabeled inference: files={len(paths)} device={device} image_size={image_size} "
        f"threshold={threshold:.3f} TTA={not args.disable_tta}",
        flush=True,
    )
    relative_paths, probabilities = predict_probabilities(
        model=model,
        input_dir=input_dir,
        paths=paths,
        image_size=image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        amp_dtype=amp_dtype,
        use_tta=not args.disable_tta,
    )

    labels = (probabilities >= threshold).astype(np.int64)
    confidences = np.maximum(probabilities, 1.0 - probabilities)
    entropies = binary_entropy(probabilities)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in (0, 1):
        (output_dir / str(label)).mkdir()

    csv_path = output_dir / "predictions.csv"
    label_counts = {"0": 0, "1": 0}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_relative_path",
            "predicted_class",
            "probability_class_1",
            "confidence",
            "entropy_nats",
            "threshold",
            "output_relative_path",
        ])
        for relative_path, label, probability, confidence, entropy in zip(
            relative_paths, labels, probabilities, confidences, entropies
        ):
            source = input_dir / relative_path
            destination = output_dir / str(int(label)) / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            writer.writerow([
                relative_path,
                int(label),
                float(probability),
                float(confidence),
                float(entropy),
                threshold,
                destination.relative_to(output_dir).as_posix(),
            ])
            label_counts[str(int(label))] += 1

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "samples": len(relative_paths),
        "class_counts": label_counts,
        "threshold": threshold,
        "image_size": image_size,
        "tta": not args.disable_tta,
        "mean_probability_class_1": float(probabilities.mean()),
        "mean_confidence": float(confidences.mean()),
        "mean_entropy_nats": float(entropies.mean()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
