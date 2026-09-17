# Data interface

## Dataset download

[Point-cloud dataset and labels](https://drive.google.com/drive/folders/1aRxrWtS6Yc7aUzaIk92wMO2Zwj5C_ETQ?usp=sharing)

Place the downloaded files under `data/`, preserving the structure below, or update the corresponding paths in `config/`.

## Point-cloud archives

The processed point clouds are stored as `data/pointcloud_1024/*.npz`. Each archive may contain one or more samples and provides:

- `sample_id`: unique string identifier, shape `[N]`
- `xyz`: centered point cloud, shape `[N, 1024, 3]`, `float32`
- `date`: observation-date label, shape `[N]`
- `DAT`: days after transplanting, shape `[N]`
- `source_area`: field-area identifier, shape `[N]`

Raw per-plant LAS/LAZ files can be converted with `01_pointcloud_preprocessing.py`. The preprocessing manifest, `data/pointcloud_manifest.csv`, contains:

```text
sample_id,source_las,date,DAT,source_area
```

Each `sample_id` uniquely identifies a plant-date observation. Including the source area, date, plot, and plant identifiers in this field avoids collisions between identically named source files.

## Trait labels

`data/manual_traits.csv` contains one row per plant-date observation:

```text
sample_id,plant_uid,variety_id,source_area,plot_id,plant_id,date,DAT,manual_tiller,manual_leaf_age,maturity_productive_tiller,yield_g_per_plant,group
```

`plant_uid` remains constant across observation dates, whereas `sample_id` identifies one point cloud. `maturity_productive_tiller` is the plant-specific productive-panicle count recorded at maturity.

## Data splits

`data/splits.csv` is the authoritative split table:

```text
sample_id,split,entity_type
```

Accepted `split` values are `train`, `internal_validation`, and `held_out_replicate_test`. Point-cloud rows use the plant-date `sample_id`. Temporal-sequence rows use `plant_uid` with `entity_type=temporal_sequence`. The held-out test samples originate from the spatially independent replicate area.

## Multimodal inputs

`data/model_inputs.csv` contains one row per plant and observation date:

```text
plant_uid,variety_id,source_area,plot_id,plant_id,pointcloud_sample_id,split,date,DAT,NDVI,LCI,GNDVI,OSAVI,NDRE,canopy_coverage,plant_height,predicted_tiller,predicted_leaf_age,vigor_score
```

The expected DAT values are `0,4,8,12,16,20,24,28,32,36`.

`predicted_tiller` and `predicted_leaf_age` are derived first-stage model outputs distributed with the dataset. They are not manual labels, and this repository does not automatically merge the per-run `03_train_pointcloud.py` prediction files into `model_inputs.csv`. The released columns were generated before multimodal training under the following conditions:

- separate PointNet + DAT regressors were fitted for tiller number and leaf age;
- each input used a deterministically sampled, centroid-centered 1,024-point cloud and standardized DAT late fusion;
- the main-area training split was used for parameter fitting, the internal-validation split was used for checkpoint selection and early stopping, and the spatially independent replicate area was used only for inference;
- optimization and stopping settings followed `config/pointcloud.yaml`; and
- predictions were joined to temporal rows by `pointcloud_sample_id` and then supplied unchanged to the multimodal models.

`08_target_reconstruction_controls.py` separately recomputes seed-specific XYZ-only PointNet predictions for the control experiments and writes them to `outputs/multimodal/target_controls/seed_<seed>/pointnet_derived_inputs.csv`. Those control outputs do not replace the distributed `model_inputs.csv` columns.

## Multispectral feature construction

The reference feature implementation uses pixels with `NDVI >= 0.45` as the vegetation mask. `canopy_coverage` is the fraction of finite pixels meeting this threshold, and the five spectral indices in `model_inputs.csv` are the means over the same masked pixels. The calculations are implemented in `feature_engineering.py`.

## Reference vigor target

`data/reference_vigor.csv` contains one row per monitored plant:

```text
plant_uid,plant_id,variety_id,plot_id,source_area,group,a_days,b_days,ax,bx,maturity_productive_tiller,threshold_attained,yield_g_per_plant,normalized_a,normalized_b,vigor_score
```

Generate this table from the manual measurements with:

```bash
python 05_vigor_score.py --input data/manual_traits.csv --output data/reference_vigor.csv
```

`ax` and `bx` are aliases of `a_days` and `b_days`, respectively. `threshold_attained` records whether the maturity-derived productive-tiller threshold was reached within the observed trajectory. The normalization limits are calculated across the supplied reference population, following the vigor-proxy definition used in the manuscript.
