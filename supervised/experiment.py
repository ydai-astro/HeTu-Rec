#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import BinaryImageDataset, ModeratePositiveBatchSampler, make_transform
from engine import predict_csv
from metrics import compute_metrics, find_best_threshold
from model import build_model, parameter_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Data", type=Path)
    parser.add_argument("--dinov3-repo", default="../external/dinov3", type=Path)
    parser.add_argument(
        "--weights",
        default="../weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        type=Path,
    )
    parser.add_argument("--out-dir", default="runs/full_d4_seed42", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--train-batch-size", default=4, type=int)
    parser.add_argument("--eval-batch-size", default=16, type=int)
    parser.add_argument("--gradient-accumulation", default=8, type=int)
    parser.add_argument("--max-steps", default=30000, type=int)
    parser.add_argument("--warmup-steps", default=1000, type=int)
    parser.add_argument("--eval-every", default=500, type=int)
    parser.add_argument("--early-stopping-patience", default=12, type=int)
    parser.add_argument("--backbone-lr", default=3e-6, type=float)
    parser.add_argument("--head-lr", default=5e-4, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--label-smoothing", default=0.01, type=float)
    parser.add_argument(
        "--sampling",
        choices=["moderate_positive", "natural"],
        default="moderate_positive",
    )
    parser.add_argument("--positive-weight", default=1.0, type=float)
    parser.add_argument("--disable-d4", action="store_true")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimizer_groups(model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        part = "backbone" if name.startswith("backbone.") else "head"
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        key = (part, decay)
        groups.setdefault(key, []).append(parameter)
    result = []
    for (part, decay), parameters in groups.items():
        result.append({
            "params": parameters,
            "lr": backbone_lr if part == "backbone" else head_lr,
            "weight_decay": weight_decay if decay else 0.0,
            "group_name": f"{part}_{'decay' if decay else 'no_decay'}",
        })
    return result


def scheduler_lambda(step: int, warmup: int, maximum: int) -> float:
    if step < warmup:
        return max(step, 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(maximum - warmup, 1))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, model, optimizer, scheduler, args, step, metric, threshold):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "selection_metric": "positive_f1",
        "selection_metric_value": metric,
        "validation_threshold": threshold,
        "model_config": {
            "dropout": args.dropout,
            "image_size": args.image_size,
            "model_name": "dinov3_vitl16",
            "weights_name": args.weights.name,
        },
        "training_args": vars(args),
    }, path)


def main() -> None:
    args = parse_args()
    if args.image_size % 16:
        raise ValueError("DINOv3 ViT-L/16 image size must be divisible by 16.")
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    print("Experiment goal: fully fine-tune DINOv3 ViT-L/16 for binary classification.")
    print(f"Device={device} sampling={args.sampling} D4={not args.disable_d4}")
    model = build_model(args.dinov3_repo, args.weights, dropout=args.dropout).to(device)
    summary = parameter_summary(model)
    print(
        f"Parameters total={summary['total']:,} trainable={summary['trainable']:,} "
        f"trainable_percent={summary['trainable_percent']:.3f}"
    )

    train_csv = args.data_root / "splits" / "train.csv"
    val_csv = args.data_root / "splits" / "val.csv"
    train_dataset = BinaryImageDataset(
        args.data_root,
        train_csv,
        make_transform(args.image_size, train=True, d4=not args.disable_d4),
    )
    if args.sampling == "moderate_positive":
        batch_sampler = ModeratePositiveBatchSampler(
            train_dataset.labels,
            batch_size=args.train_batch_size,
            positive_fraction=0.125,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
    else:
        batch_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

    groups = optimizer_groups(model, args.backbone_lr, args.head_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_lambda(step, args.warmup_steps, args.max_steps),
    )
    class_weights = torch.tensor([1.0, args.positive_weight], device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )

    start_step = 0
    best_f1 = -1.0
    stale_evaluations = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])
        best_f1 = float(checkpoint["selection_metric_value"])
        print(f"Resumed from {args.resume} at step={start_step} best_f1={best_f1:.5f}")

    history_path = args.out_dir / "history.csv"
    if not history_path.exists() or start_step == 0:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                "step", "train_loss", "val_threshold", "val_accuracy", "val_pr_auc",
                "val_macro_f1", "val_positive_precision", "val_positive_recall", "val_positive_f1",
            ])

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    epoch = 0
    running_loss = 0.0
    running_batches = 0
    micro_step = 0
    stop = False

    while global_step < args.max_steps and not stop:
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if device.type == "cuda" else torch.float32,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            running_loss += loss.item()
            running_batches += 1
            micro_step += 1

            if micro_step % args.gradient_accumulation != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"Training step={global_step}/{args.max_steps} "
                    f"loss={running_loss / max(running_batches, 1):.5f} "
                    f"backbone_lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )

            if global_step % args.eval_every == 0 or global_step == args.max_steps:
                labels_np, probabilities, _ = predict_csv(
                    model,
                    args.data_root,
                    val_csv,
                    args.image_size,
                    args.eval_batch_size,
                    args.workers,
                    device,
                    amp_dtype,
                    tta=False,
                )
                threshold = find_best_threshold(labels_np, probabilities)
                metrics = compute_metrics(labels_np, probabilities, threshold)
                train_loss = running_loss / max(running_batches, 1)
                positive = metrics["class_1"]
                print(
                    f"Validation step={global_step} threshold={threshold:.3f} "
                    f"accuracy={metrics['accuracy']:.5f} pr_auc={metrics['pr_auc']:.5f} "
                    f"macro_f1={metrics['macro_f1']:.5f} positive_f1={positive['f1']:.5f}",
                    flush=True,
                )
                with history_path.open("a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow([
                        global_step, train_loss, threshold, metrics["accuracy"], metrics["pr_auc"],
                        metrics["macro_f1"], positive["precision"], positive["recall"], positive["f1"],
                    ])
                if positive["f1"] > best_f1:
                    best_f1 = float(positive["f1"])
                    stale_evaluations = 0
                    save_checkpoint(
                        args.out_dir / "best_checkpoint.pt",
                        model, optimizer, scheduler, args, global_step, best_f1, threshold,
                    )
                    print(f"Saved new best checkpoint at step={global_step}")
                else:
                    stale_evaluations += 1
                save_checkpoint(
                    args.out_dir / "last_checkpoint.pt",
                    model, optimizer, scheduler, args, global_step, best_f1, threshold,
                )
                running_loss = 0.0
                running_batches = 0
                model.train()
                if stale_evaluations >= args.early_stopping_patience:
                    print("Early stopping: validation positive F1 did not improve.")
                    stop = True
                    break
            if global_step >= args.max_steps:
                break
        epoch += 1

    final_info = {
        "best_validation_positive_f1": best_f1,
        "final_step": global_step,
        "best_checkpoint": str((args.out_dir / "best_checkpoint.pt").resolve()),
        "parameter_summary": summary,
        "test_was_not_used_during_training": True,
    }
    (args.out_dir / "final_info.json").write_text(
        json.dumps(final_info, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_info, indent=2))


if __name__ == "__main__":
    main()
