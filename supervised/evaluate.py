#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from engine import predict_csv
from metrics import compute_metrics, find_best_threshold, save_evaluation
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--data-root", default="Data", type=Path)
    parser.add_argument("--dinov3-repo", default="../external/dinov3", type=Path)
    parser.add_argument(
        "--weights",
        default="../weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--calibrate-threshold", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--eval-batch-size", default=16, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = build_model(args.dinov3_repo, args.weights, dropout=float(config["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    csv_path = args.data_root / "splits" / f"{args.split}.csv"
    labels, probabilities, paths = predict_csv(
        model,
        args.data_root,
        csv_path,
        int(config["image_size"]),
        args.eval_batch_size,
        args.workers,
        device,
        amp_dtype,
        tta=args.tta,
    )
    if args.calibrate_threshold:
        if args.split != "val":
            raise ValueError("Threshold calibration is allowed only on the validation split.")
        threshold = find_best_threshold(labels, probabilities)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "threshold.json").write_text(
            json.dumps({"threshold": threshold, "tta": args.tta}, indent=2),
            encoding="utf-8",
        )
    elif args.threshold_file:
        threshold = float(json.loads(args.threshold_file.read_text(encoding="utf-8"))["threshold"])
    else:
        threshold = float(checkpoint["validation_threshold"])

    metrics = compute_metrics(labels, probabilities, threshold)
    save_evaluation(args.output_dir, args.split, paths, labels, probabilities, metrics)
    print(
        f"{args.split} accuracy={metrics['accuracy']:.5f} pr_auc={metrics['pr_auc']:.5f} "
        f"macro_f1={metrics['macro_f1']:.5f} positive_f1={metrics['class_1']['f1']:.5f} "
        f"threshold={threshold:.3f} TTA={args.tta}"
    )


if __name__ == "__main__":
    main()
