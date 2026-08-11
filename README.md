#SNR-HML
Reproducible code for two linked image-analysis workflows:

1. **Unsupervised learning (`unsupervised/`)**: extract DINOv3 image features,
   reduce their dimensionality with UMAP, and inspect clusters from K-means,
   agglomerative clustering, and Birch.
2. **Supervised learning (`supervised/`)**: fully fine-tune DINOv3 ViT-L/16 for
   imbalanced binary PNG classification, with D4 rotation/reflection handling,
   validation-only threshold calibration, and independent test evaluation.

Private images, labels, model weights, checkpoints, and generated predictions
are not distributed in this repository. Every command below uses paths relative
to the repository root.

## Repository Layout

```text
.
├── data/                         # Place private raw images here
├── external/                     # Automatically cloned DINOv3 source (ignored)
├── weights/                      # Downloaded pretrained weights (ignored)
├── outputs/                      # Features, embeddings, clusters (ignored)
├── scripts/                      # Environment, hardware, and weight helpers
├── unsupervised/                 # Feature extraction, UMAP, clustering
└── supervised/                   # Binary DINOv3 fine-tuning workflow
```

## Environment and Device Setup

Python 3.10+ and a CUDA-capable NVIDIA GPU are recommended. The reference
environment used PyTorch 2.7.1 with CUDA 11.8. Install the PyTorch wheel matching
your local CUDA driver; for CUDA 11.8:

```bash
conda env create -f SNR-HML
.yml
conda activate SNR-HML


pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
bash scripts/setup_environment.sh
bash scripts/download_dinov3_vitl16.sh
```

`scripts/setup_environment.sh` installs project dependencies, clones the official
[DINOv3](https://github.com/facebookresearch/dinov3) repository to
`external/dinov3/`, and reports PyTorch/CUDA/GPU availability. The 1.13 GB
ViT-L/16 LVD-1689M checkpoint is downloaded from Meta's official DINOv3 release
into `weights/` and is excluded from Git.

## Unsupervised Workflow

Place source PNG files below `data/raw_images/`. Then follow the detailed
[unsupervised guide](unsupervised/README.md):

```bash
python -m unsupervised.feature_encoding.extract_features \
  --input-dir data/raw_images \
  --output outputs/features/dinov3_vitl16_cls_patch_l2.tsv

python -m unsupervised.dimensionality_reduction.select_umap_dimension \
  --input outputs/features/dinov3_vitl16_cls_patch_l2.tsv \
  --output outputs/metrics/umap_db_scores.csv \
  --dimensions 50 100 150 200 250 300 \
  --clusters 50

python -m unsupervised.dimensionality_reduction.run_umap \
  --input outputs/features/dinov3_vitl16_cls_patch_l2.tsv \
  --output outputs/embeddings/umap_100.tsv \
  --components 100

python -m unsupervised.clustering.cluster_features \
  --input outputs/embeddings/umap_100.tsv \
  --output-dir outputs/clusters/umap_100 \
  --image-root data/raw_images \
  --clusters 50
```

The unsupervised stage is exploratory. Davies-Bouldin scores and clusters are not
ground-truth classification metrics; visual review and scientific validation are
required before assigning semantic labels.

## Supervised Workflow

The binary workflow has its own private data root, `supervised/Data/`. From the
repository root, prepare numbered image folders using the reference positive set
`18, 28, 31` (replace these IDs as appropriate):

```bash
cd supervised

python prepare_dataset.py \
  --source ../../your_source_directory \
  --positive-folders 18 28 31 \
  --num-folders 50 \
  --image-size 100 \
  --seed 42

python verify_dataset.py
```

The script performs a deterministic source-folder-wise 8:1:1 train/validation/test
split. It does not use the optional `rest/` folder in any model-selection step.

Train, calibrate, and independently test:

```bash
nohup bash run_train.sh > full_d4_seed42.log 2>&1 &
nohup bash run_test.sh runs/full_d4_seed42 > full_d4_seed42_test.log 2>&1 &
```

The model resizes 100x100 RGB PNGs to 224x224 in memory using bicubic
interpolation without cropping. Training applies a random D4 transform. Validation
and test use 8-view D4 test-time augmentation. The threshold is selected only on
the validation split and locked before the independent test split is evaluated.

For unlabeled inference after testing:

```bash
nohup bash run_label_rest.sh > label_rest.log 2>&1 &
```

Predictions are written to `supervised/Data/not_label/`; this directory is ignored
by Git.

## Reference Binary Configuration

| Component | Setting |
|---|---|
| Backbone | DINOv3 ViT-L/16, full fine-tuning |
| Pooling/head | CLS token + mean patch token, MLP, dropout 0.1 |
| Optimizer | AdamW |
| Learning rates | Backbone 3e-6; head 5e-4 |
| Effective batch size | 32 (batch 4, accumulation 8) |
| Positive exposure | 12.5% per effective batch |
| Selection metric | Validation positive-class F1 |
| Test-time inference | Validation-calibrated threshold, 8-view D4 TTA |

## Reproducibility

- Never use the independent test split for hyperparameter tuning or threshold
  selection.
- Record the data split, seed, image size, model weight version, and all metrics.
- For a final paper result, repeat the final setting across multiple seeds or
  folds and report uncertainty, not only a single best run.

## License and Third-Party Dependencies

Project code is released under the [MIT License](LICENSE). DINOv3 source code and
weights are external dependencies governed by their own official license terms.
