#!/usr/bin/env python3
import importlib

import torch
import torchvision


packages = ("numpy", "PIL", "sklearn", "umap", "timm", "astropy")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for package in packages:
    importlib.import_module(package)
print("Python dependencies: OK")
