# Dataset

The PlantVillage dataset is not committed to this repository due to its size.

## Download Instructions

1. Go to Kaggle: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage
2. Download and unzip the archive.
3. Place the contents so the directory structure looks like:

```
data/
└── raw/
    └── PlantVillage/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        ├── Apple___Cedar_apple_rust/
        ├── Apple___healthy/
        ├── ...
        ├── Potato___Early_blight/
        ├── Potato___Late_blight/
        ├── Potato___healthy/
        ├── Tomato___Bacterial_spot/
        └── ...
```

Each sub-directory is named `<Plant>___<Condition>` and contains `.jpg` images.

## Configuration

The `data_dir` field in each YAML config points to this folder.
Default value: `data/raw/PlantVillage`

You can override it at runtime:

```bash
# Override data_dir via config edit, or pass a custom config
python -m src.data --config configs/baseline_resnet18_potato.yaml
```

The scripts also work when the dataset lives elsewhere — just update
`data_dir` in the relevant YAML config file.

## Subset Mode

For fast initial experiments the configs use `subset_prefix: "Potato"`,
which limits training to the three Potato classes:

- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`

Set `subset_prefix: null` (or remove the key) to use all 38 classes.
