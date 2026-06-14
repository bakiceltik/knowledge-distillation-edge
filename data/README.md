# Datasets

Neither dataset is committed to this repository due to size. Download them from
Kaggle and place them under `data/raw/` as described below.

| Dataset | Role in this project | Classes | Source |
|---|---|---|---|
| **Cassava Leaf Disease** | Main study (5-fold CV) | 5 | https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification |
| **PlantVillage** | Preliminary single-split demo | 38 | https://www.kaggle.com/datasets/mohitsingh1804/plantvillage |

## Cassava (main study)

1. Download and unzip the [Cassava Leaf Disease](https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification)
   dataset from Kaggle.
2. Arrange it as one sub-directory per class (no train/val/test split — the code
   creates reproducible stratified splits itself):

```
data/
└── raw/
    └── cassava/
        ├── Cassava___bacterial_blight/
        ├── Cassava___brown_streak_disease/
        ├── Cassava___green_mottle/
        ├── Cassava___healthy/
        └── Cassava___mosaic_disease/
```

Each sub-directory contains `.jpg` images. This is the layout the
`*_cassava.yaml` and `cv_*_cassava.yaml` configs expect (`data_dir: data/raw/cassava`).

## PlantVillage (preliminary study)

1. Download and unzip the [PlantVillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
   dataset from Kaggle.
2. Arrange it so the 38 class folders live under a `train/` directory:

```
data/
└── raw/
    └── PlantVillage/
        └── train/
            ├── Apple___Apple_scab/
            ├── Apple___Black_rot/
            ├── ...
            ├── Tomato___healthy/
            └── ...
```

Each sub-directory is named `<Plant>___<Condition>` and contains `.jpg` images.
The preliminary configs point `data_dir` at `data/raw/PlantVillage/train`.

## Splits and reproducibility

The code does **not** rely on a pre-existing train/val/test split. Given a flat
`<class>/` directory, [`src/data/data_utils.py`](../src/data/data_utils.py)
builds the splits deterministically from the config `seed`:

- `get_dataloaders(...)` — a single stratified 70/15/15 train/val/test split
  (used by the single-split runners and the quantization/error analysis).
- `get_cv_dataloaders(...)` — one stratified k-fold partition per fold (used by
  the cross-validation runner).

Both are seeded, so every model sees identical splits — a fair, paired
comparison.

## Configuration

The `data_dir` field in each YAML config points to the dataset root. Override it
by editing the config or passing a different one. An optional `subset_prefix`
field restricts training to classes whose folder name starts with the given
prefix (e.g. for quick smoke tests on a subset of PlantVillage); set it to
`null` to use all classes, which is what the reported experiments do.
