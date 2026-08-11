# Unsupervised Pipeline

The unsupervised workflow is intentionally separated from the supervised
classifier. It creates feature embeddings, optionally selects a UMAP dimension,
then clusters the embedding with K-means, agglomerative clustering, and/or Birch.

All commands are run from the repository root and use relative paths.

## 1. Extract DINOv3 Features

```bash
python -m unsupervised.feature_encoding.extract_features \
  --input-dir data/raw_images \
  --output outputs/features/dinov3_vitl16_cls_patch_l2.tsv \
  --feature-mode cls_patch
```

The `cls_patch` option independently L2-normalizes the CLS token and the mean
patch token, concatenates them, and L2-normalizes the resulting 2,048-dimensional
vector.

## 2. Select a UMAP Dimension

```bash
python -m unsupervised.dimensionality_reduction.select_umap_dimension \
  --input outputs/features/dinov3_vitl16_cls_patch_l2.tsv \
  --output outputs/metrics/umap_db_scores.csv \
  --dimensions 50 100 150 200 250 300 \
  --clusters 50
```

The script reports the Davies-Bouldin score for each candidate. Lower values
indicate more compact, better separated clusters for the selected K-means proxy;
they do not constitute ground-truth classification accuracy.

## 3. Run UMAP or PCA

```bash
python -m unsupervised.dimensionality_reduction.run_umap \
  --input outputs/features/dinov3_vitl16_cls_patch_l2.tsv \
  --output outputs/embeddings/umap_100.tsv \
  --components 100
```

PCA is available as a deterministic linear alternative:

```bash
python -m unsupervised.dimensionality_reduction.run_pca \
  --input outputs/features/dinov3_vitl16_cls_patch_l2.tsv \
  --output outputs/embeddings/pca_100.tsv \
  --components 100
```

## 4. Cluster and Export Assignments

```bash
python -m unsupervised.clustering.cluster_features \
  --input outputs/embeddings/umap_100.tsv \
  --output-dir outputs/clusters/umap_100 \
  --image-root data/raw_images \
  --clusters 50 \
  --methods kmeans agglomerative birch
```

Each method writes an `assignments.csv`. When `--image-root` is supplied, it also
copies the original images into `clusters/<cluster-id>/` for visual inspection.
The copied files are generated outputs and are ignored by Git.

## 5. Optional Conservative Consensus

The following keeps only images where the non-anchor methods agree with the
anchor K-means cluster after each method's clusters are majority-aligned to the
anchor. This is conservative: disagreement is treated as uncertain, not forced
into a class.

```bash
python -m unsupervised.clustering.consensus_intersection \
  --anchor outputs/clusters/umap_100/kmeans/assignments.csv \
  --others outputs/clusters/umap_100/agglomerative/assignments.csv \
           outputs/clusters/umap_100/birch/assignments.csv \
  --output outputs/consensus/assignments.csv \
  --image-root data/raw_images
```

After visual/scientific review, create a JSON file such as
`{"0": 3, "1": 1}` mapping cluster IDs to final class labels and export
filename-prefixed images:

```bash
python -m unsupervised.clustering.export_manual_labels \
  --assignments outputs/consensus/assignments.csv \
  --cluster-labels outputs/consensus/cluster_labels.json \
  --image-root data/raw_images \
  --output-dir outputs/manual_labels
```
