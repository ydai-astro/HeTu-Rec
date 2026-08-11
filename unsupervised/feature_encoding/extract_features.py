#!/usr/bin/env python3
"""Extract L2-normalized DINOv3 features from a directory of PNG images."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from unsupervised.feature_io import write_feature_tsv


class ImageDataset(Dataset):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw_images", type=Path)
    parser.add_argument("--output", default="outputs/features/dinov3_vitl16_cls_patch_l2.tsv", type=Path)
    parser.add_argument("--dinov3-repo", default="external/dinov3", type=Path)
    parser.add_argument("--weights", default="weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", type=Path)
    parser.add_argument("--feature-mode", choices=("cls", "patch_mean", "cls_patch"), default="cls_patch")
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    repo = args.dinov3_repo.resolve()
    weights = args.weights.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 repository not found: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weight not found: {weights}")

    paths = sorted(path for path in input_dir.rglob("*.png") if path.is_file())
    if not paths:
        raise RuntimeError(f"No PNG files found under: {input_dir}")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ImageDataset(input_dir, paths, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    model = torch.hub.load(str(repo), "dinov3_vitl16", source="local", pretrained=True, weights=str(weights))
    model.eval().to(device)
    names, batches = [], []
    print(f"Extracting {len(dataset)} images on {device}; feature_mode={args.feature_mode}", flush=True)
    with torch.inference_mode():
        for images, batch_names in tqdm(loader, desc="Encoding"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                features = model.forward_features(images)
                cls = features["x_norm_clstoken"]
                patch_mean = features["x_norm_patchtokens"].mean(dim=1)
                if args.feature_mode == "cls":
                    output = functional.normalize(cls, dim=1)
                elif args.feature_mode == "patch_mean":
                    output = functional.normalize(patch_mean, dim=1)
                else:
                    output = functional.normalize(torch.cat((functional.normalize(cls, dim=1), functional.normalize(patch_mean, dim=1)), dim=1), dim=1)
            names.extend(batch_names)
            batches.append(output.float().cpu().numpy())

    matrix = np.concatenate(batches, axis=0)
    write_feature_tsv(args.output, names, matrix)
    print(f"Saved features: {args.output} samples={len(names)} dimensions={matrix.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
