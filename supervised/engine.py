from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BinaryImageDataset, make_transform


@torch.no_grad()
def predict_once(model, loader, device, amp_dtype):
    model.eval()
    probabilities = []
    labels = []
    paths = []
    use_amp = device.type == "cuda"
    for images, batch_labels, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype if use_amp else torch.float32,
            enabled=use_amp,
        ):
            logits = model(images)
        probabilities.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
        labels.extend(batch_labels.tolist())
        paths.extend(batch_paths)
    return np.asarray(labels), np.asarray(probabilities), paths


@torch.no_grad()
def predict_csv(
    model,
    data_root: Path,
    csv_path: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device,
    amp_dtype,
    tta: bool,
):
    operations = range(8) if tta else [None]
    accumulated = None
    reference_labels = None
    reference_paths = None
    for operation in operations:
        transform = make_transform(
            image_size=image_size,
            train=False,
            d4=False,
            tta_index=operation,
        )
        dataset = BinaryImageDataset(data_root, csv_path, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        labels, probabilities, paths = predict_once(model, loader, device, amp_dtype)
        if reference_labels is None:
            reference_labels = labels
            reference_paths = paths
            accumulated = probabilities
        else:
            if not np.array_equal(reference_labels, labels) or reference_paths != paths:
                raise RuntimeError("Evaluation order changed between D4 passes.")
            accumulated += probabilities
    assert accumulated is not None and reference_labels is not None and reference_paths is not None
    accumulated /= len(list(operations))
    return reference_labels, accumulated, reference_paths
