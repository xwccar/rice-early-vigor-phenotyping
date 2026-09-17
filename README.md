# Rice Early Vigor Methods

Reference implementation accompanying the manuscript *Estimation of a phenology-derived early growth vigor proxy in rice using time-series UAV LiDAR and multispectral phenotyping*.

## Data

[Point-cloud dataset and labels](https://drive.google.com/drive/folders/1aRxrWtS6Yc7aUzaIk92wMO2Zwj5C_ETQ?usp=sharing)

## Repository structure

| Method component | Implementation |
| --- | --- |
| Individual-plant fixed-point preprocessing | `01_pointcloud_preprocessing.py` |
| PointNet, PointNet++, DGCNN, PCT, and Point Transformer | `02_pointcloud_models/` |
| Point-cloud trait regression and repeated-seed comparison | `03_train_pointcloud.py` |
| DAT-only, PointNet-only, and PointNet + DAT ablation | `04_dat_ablation.py` |
| Phenology-derived vigor proxy | `05_vigor_score.py` |
| Spectral indices and canopy coverage | `feature_engineering.py` |
| Dual-branch LSTM-attention model with OGR weighting | `06_multimodal_model.py` and `multimodal_core.py` |
| XGBoost, GRU, TCN, Transformer, early-fusion LSTM, and uniform-weighting comparisons | `07_baselines.py` and `06_multimodal_model.py --strategy uniform` |
| Target-reconstruction and input-dependence controls | `08_target_reconstruction_controls.py` |
| Paired model tests and vigor-yield analyses | `09_statistics.py` |
| Consolidation of repeated-run metrics | `10_collect_results.py` |

## Environment

The reference environment uses Python 3.9 and PyTorch 2.5.

```bash
conda env create -f environment.yml
conda activate rice-vigor-methods
```

Alternatively, install `requirements.txt` in an existing Python environment. Use the PyTorch build compatible with the local CUDA driver.

## Experimental protocol

### Point-cloud trait models

- Tiller number and leaf age are fitted as separate regression targets.
- Each plant is represented by 1,024 XYZ points after centroid subtraction, without scale normalization or point augmentation.
- Ten observation dates are used: DAT 0, 4, 8, 12, 16, 20, 24, 28, 32, and 36.
- Ten paired random seeds (41-50) are used for repeated comparisons.
- Optimization uses AdamW with learning rate `1e-3`, weight decay `1e-4`, and cosine annealing to `1e-5`.
- The batch size is 8, the maximum number of epochs is 200, and the gradient-norm clipping threshold is 1.0.
- Early stopping begins after 40 epochs and uses a patience of 30 epochs on the internal-validation loss.
- PointNet feature-transform orthogonality regularization uses a coefficient of `0.001`.
- DGCNN uses `k=20`; PCT and Point Transformer use `k=16`.
- Training uses FP32 precision without automatic mixed precision.

The proposed point-cloud configuration uses PointNet features with standardized DAT late fusion. The same split and seed are retained when comparing PointNet, PointNet++, DGCNN, PCT, and Point Transformer.

### Multimodal vigor model

- Six multispectral inputs and three structural inputs are arranged as ten-date sequences.
- Multispectral indices and canopy coverage use an `NDVI >= 0.45` vegetation mask.
- The spectral and structural branches use separate LSTM-attention encoders with hidden size 16 and representation size 16.
- The fusion head uses hidden dimensions 32 and 16 with dropout `0.15`.
- Training uses SGD with learning rate `1e-3`, momentum 0, weight decay `1e-4`, batch size 16, and gradient-norm clipping at 1.0.
- The maximum number of epochs is 200; early stopping begins after 40 epochs with patience 30.
- Three supervised losses are combined using inverse overfitting-to-generalization ratio weights updated once per epoch (`epsilon=1e-8`).
- The same ten seeds (41-50) are used for multimodal comparisons.

### Comparison models

- XGBoost uses 1,500 trees, maximum depth 3, learning rate `0.02`, minimum child weight 2, row subsampling `0.8`, and column subsampling `0.8`.
- GRU uses a 48-unit recurrent layer followed by LayerNorm and a 48-24-1 regression head.
- TCN uses 48 channels with dilation rates 1, 2, and 4, followed by a 48-24-1 regression head.
- Transformer uses a 32-dimensional projection, two encoder layers, four attention heads, feed-forward dimension 128, mean pooling, LayerNorm, and a 32-16-1 regression head.
- Early-fusion LSTM uses a 48-unit recurrent layer, additive attention, and a 48-24-1 regression head.

## Running the workflow

Run all commands from the repository root.

```bash
# Convert per-plant LAS/LAZ files into deterministic 1,024-point archives
python 01_pointcloud_preprocessing.py \
  --manifest data/pointcloud_manifest.csv \
  --output-dir data/pointcloud_1024

# Point-cloud traits
python 03_train_pointcloud.py --target tiller --all-models --all-seeds
python 03_train_pointcloud.py --target leaf_age --all-models --all-seeds

# DAT ablation
python 04_dat_ablation.py --all-seeds

# Reference vigor proxy and proposed multimodal model
python 05_vigor_score.py --input data/manual_traits.csv --output data/reference_vigor.csv
python 06_multimodal_model.py --strategy ogr --all-seeds

# Comparison models and controls
python 06_multimodal_model.py --strategy uniform --all-seeds
python 07_baselines.py --all-seeds
python 08_target_reconstruction_controls.py --all-seeds

# Aggregate runs, paired tests, and biological analyses
python 10_collect_results.py
python 09_statistics.py
```

The main-area training and internal-validation samples are used for model fitting and checkpoint selection. Samples from the spatially independent replicate area are reserved for held-out testing. Seeds are paired across model comparisons.

## Verification

```bash
python tests/smoke_test.py
```

The verification suite checks preprocessing, all five point-cloud backbones, DAT fusion, vigor-score calculation, spectral-index calculation, temporal models, and OGR weighting with compact, self-contained fixtures.
