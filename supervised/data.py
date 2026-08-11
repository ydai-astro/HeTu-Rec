from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomD4:
    def __call__(self, image: Image.Image) -> Image.Image:
        operation = random.randrange(8)
        rotation = operation % 4
        if rotation == 1:
            image = image.transpose(Image.Transpose.ROTATE_90)
        elif rotation == 2:
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif rotation == 3:
            image = image.transpose(Image.Transpose.ROTATE_270)
        if operation >= 4:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class FixedD4:
    def __init__(self, operation: int):
        if operation not in range(8):
            raise ValueError("D4 operation must be in [0, 7].")
        self.operation = operation

    def __call__(self, image: Image.Image) -> Image.Image:
        rotation = self.operation % 4
        if rotation == 1:
            image = image.transpose(Image.Transpose.ROTATE_90)
        elif rotation == 2:
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif rotation == 3:
            image = image.transpose(Image.Transpose.ROTATE_270)
        if self.operation >= 4:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


def make_transform(image_size: int, train: bool, d4: bool = True, tta_index: int | None = None):
    operations = []
    if tta_index is not None:
        operations.append(FixedD4(tta_index))
    elif train and d4:
        operations.append(RandomD4())
    operations.extend([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(operations)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty split CSV: {csv_path}")
    return rows


class BinaryImageDataset(Dataset):
    def __init__(self, data_root: Path, csv_path: Path, transform):
        self.data_root = data_root
        self.rows = read_rows(csv_path)
        self.transform = transform
        self.labels = [int(row["label"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.data_root / row["relative_path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"]), row["relative_path"]


class ModeratePositiveBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        positive_fraction: float = 0.125,
        seed: int = 42,
    ):
        if batch_size < 2:
            raise ValueError("Batch size must be at least 2.")
        self.positive = [index for index, label in enumerate(labels) if label == 1]
        self.negative = [index for index, label in enumerate(labels) if label == 0]
        if not self.positive or not self.negative:
            raise ValueError("Both classes must be present in the training split.")
        self.batch_size = batch_size
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be between 0 and 1.")
        # One positive sample is distributed across a short cycle of batches.
        # With batch_size=4 and positive_fraction=0.125, this is one positive
        # image followed by seven negatives across every two physical batches.
        self.cycle_batches = max(1, round(1.0 / (batch_size * positive_fraction)))
        self.actual_positive_fraction = 1.0 / (batch_size * self.cycle_batches)
        self.seed = seed
        self.epoch = 0
        self.num_batches = math.ceil(len(labels) / batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _take(pool: list[int], count: int, cursor: int, rng: random.Random):
        selected = []
        while len(selected) < count:
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
            available = min(count - len(selected), len(pool) - cursor)
            selected.extend(pool[cursor:cursor + available])
            cursor += available
        return selected, cursor

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        positive = self.positive.copy()
        negative = self.negative.copy()
        rng.shuffle(positive)
        rng.shuffle(negative)
        positive_cursor = 0
        negative_cursor = 0
        for batch_index in range(self.num_batches):
            positive_count = 1 if batch_index % self.cycle_batches == 0 else 0
            negative_count = self.batch_size - positive_count
            pos, positive_cursor = self._take(positive, positive_count, positive_cursor, rng)
            neg, negative_cursor = self._take(negative, negative_count, negative_cursor, rng)
            batch = pos + neg
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
