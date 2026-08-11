#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.run_dir / "history.csv").open(encoding="utf-8")))
    steps = [int(row["step"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, [float(row["train_loss"]) for row in rows], color="#2f6b4f")
    axes[0].set(xlabel="Optimizer step", ylabel="Training loss", title="Training loss")
    axes[1].plot(steps, [float(row["val_positive_f1"]) for row in rows], label="Positive F1")
    axes[1].plot(steps, [float(row["val_macro_f1"]) for row in rows], label="Macro F1")
    axes[1].plot(steps, [float(row["val_pr_auc"]) for row in rows], label="PR-AUC")
    axes[1].set(xlabel="Optimizer step", ylabel="Score", ylim=(0, 1), title="Validation metrics")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.run_dir / "Figure_1_training_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, [float(row["val_threshold"]) for row in rows], color="#9b3a3a")
    ax.set(xlabel="Optimizer step", ylabel="Validation threshold", ylim=(0, 1), title="Calibrated threshold")
    fig.tight_layout()
    fig.savefig(args.run_dir / "Figure_2_threshold.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
