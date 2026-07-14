# Knowledge Distillation Across Teacher Architectures for Edge-Efficient Plant Disease Classification

A university machine-learning project on knowledge distillation (KD) for plant
disease classification. A large **teacher** network is distilled into a compact
**MobileNetV3-Small** student suitable for edge deployment. The central question
is not just *"does distillation help?"* but *"does the **architecture of the
teacher** matter?"* — so we compare three teacher families under a single,
cross-validated protocol.

## Key finding

On the Cassava leaf-disease dataset (5-fold cross-validation), distillation
lifts the student **+1.4–1.8 accuracy points** over a supervised baseline, but
**which teacher is used barely matters**: students distilled from ResNet50,
EfficientNet-B2, and ViT-B/16 land within 0.5 points of each other, despite the
teachers spanning 82–85% accuracy, two architecture families, and 7.7M–85.8M
parameters. The weakest and largest teacher (ViT-B/16) teaches as well as the
strongest — and its student even **surpasses its own teacher**. The benefit
comes from the distillation *procedure*, not the teacher's capacity or
architecture.

## Studies

1. **Main study (Cassava, 5 classes):** stratified 5-fold cross-validation, three
   teachers each retrained per fold and distilled into MobileNetV3-Small, with a
   paired significance test against the supervised student baseline and a KD
   hyperparameter sensitivity ablation.
2. **Preliminary study (PlantVillage, 38 classes):** a single-split
   ResNet50 → MobileNetV3-Small demonstration of the pipeline.

The full write-up is in [reports/report.tex](reports/report.tex).

## Datasets

| Dataset | Role | Classes | Source |
|---|---|---|---|
| **Cassava Leaf Disease** | Main study | 5 | https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification |
| **PlantVillage** | Preliminary | 38 | https://www.kaggle.com/datasets/mohitsingh1804/plantvillage |

Datasets are **not** committed (see [data/README.md](data/README.md)). Place them as
`data/raw/cassava/<class>/...` and `data/raw/PlantVillage/train/<class>/...`
(one sub-directory per class).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Device is selected automatically: CUDA > MPS > CPU. **All** training runs
reported in the paper — Cassava and PlantVillage alike — were run on a single
NVIDIA RTX 4070 (CUDA) with automatic mixed precision, so the two studies are
directly comparable. Quantization and latency are measured on CPU.

## How to run the main study (Cassava, 5-fold CV)

The cross-validation runner ([src/cross_validate.py](src/cross_validate.py)) handles
both supervised (baseline/teacher) and distillation modes — the mode is inferred
from the config (a `teacher_model_name` field selects distillation). For
distillation, the teacher is **retrained from scratch on each fold's training
split**, so the held-out test fold never leaks into the teacher.

```bash
# 1. Teachers, cross-validated (supervised)
python -m src.cross_validate --config configs/cv_teacher_resnet50_cassava.yaml
python -m src.cross_validate --config configs/cv_teacher_efficientnet_b2_cassava.yaml
python -m src.cross_validate --config configs/cv_teacher_vit_b16_cassava.yaml

# 2. Student baseline, cross-validated (no teacher) — the reference point
python -m src.cross_validate --config configs/cv_baseline_student_cassava.yaml

# 3. Distillation, cross-validated (teacher retrained per fold)
python -m src.cross_validate --config configs/cv_distillation_resnet50_cassava.yaml
python -m src.cross_validate --config configs/cv_distillation_efficientnet_b2_cassava.yaml
python -m src.cross_validate --config configs/cv_distillation_vit_b16_cassava.yaml

# 4. Collate everything into one table + paired significance tests
python -m src.collate_cv

# 5. KD temperature/alpha sensitivity ablation (single split, fold 0)
python -m src.ablate_kd --config configs/ablation_resnet50_cassava.yaml
```

`--folds N` overrides the fold count for any cross_validate run (e.g. `--folds 3`
for a quicker pass). `collate_cv` writes `outputs/cv_summary.{csv,md,tex}` and
`outputs/cv_significance.md`; the `.tex` files are `\input` directly by the
report.

## Single-split runs (Cassava and PlantVillage)

The single-split runners are [src/train_baseline.py](src/train_baseline.py)
(supervised) and [src/train_distillation.py](src/train_distillation.py)
(distillation). The `*_cassava.yaml` configs below run a single 70/15/15 split on
Cassava (the quick, non-CV counterpart to the main study):

```bash
python -m src.train_baseline     --config configs/teacher_resnet50_cassava.yaml
python -m src.train_baseline     --config configs/student_mobilenetv3_cassava.yaml
python -m src.train_distillation --config configs/distillation_resnet50_cassava.yaml
# or: bash scripts/run_cassava_{teacher,student,distillation}.sh
```

## PlantVillage (38 classes)

Note the `data_dir`: it points at the release's `train/` folder, which holds all
38 class directories. Pointing at `data/raw/PlantVillage` instead would make
`ImageFolder` read `train`/`val` as *two* classes.

```bash
python -m src.train_baseline     --config configs/teacher_resnet50_full_rtx4070.yaml
python -m src.train_baseline     --config configs/student_mobilenetv3_full_rtx4070.yaml
python -m src.train_distillation --config configs/distillation_resnet50_full_rtx4070.yaml

# both students under two further seeds (split fixed) -> the +-SD in the table below
python -m src.train_baseline     --config configs/pv_replicate/supervised_seed43.yaml
python -m src.train_baseline     --config configs/pv_replicate/supervised_seed44.yaml
python -m src.train_distillation --config configs/pv_replicate/distilled_seed43.yaml
python -m src.train_distillation --config configs/pv_replicate/distilled_seed44.yaml
```

## Quantization, latency, and analysis figures

int8 post-training quantization uses the **PT2E** (`torch.export`) API, not the
legacy FX graph mode. On torch 2.6 `quantize_fx.convert_fx` corrupts this model:
converting with the qconfig disabled everywhere — i.e. quantizing *nothing* —
still collapses test accuracy from 84% to ~15%, so any number from that path
measures the bug rather than quantization.

```bash
# int8 PTQ: static (per-channel), mixed precision (SE blocks in float32),
# per-tensor weights, and dynamic. Calibration is seeded -- without a fixed
# calibration_seed the reported int8 accuracy moves by several points per run.
python -m src.quantize --config configs/quantization_cassava.yaml
python -m src.quantize --config configs/quantization_cassava_mixed.yaml
python -m src.quantize --config configs/quantization_cassava_pertensor.yaml
python -m src.quantize --config configs/quantization_cassava_dynamic.yaml

# is the int8 collapse caused by distillation, or by the architecture?
# 5 seeds x {supervised, distilled}, matched budgets, then quantize all ten
# (each averaged over 5 calibration draws).
python -m src.replicate_quant --config configs/replicate_quant_cassava.yaml

# the four controls behind the collapse claim: ResNet50 teacher control,
# calibration-size sweep, activation-range comparison, per-layer sensitivity.
python -m src.quant_analysis --config configs/quant_analysis_cassava.yaml

# batch-one latency. RUN ON AN IDLE MACHINE: timings taken alongside training
# jobs are worthless. The script reports the spread between five independent
# repeats so an unreliable number is visible rather than hidden.
python -m src.benchmark_latency --config configs/benchmark_latency_cassava.yaml

# t-SNE of the student's penultimate embeddings on the Cassava test set
python -m src.visualize_embeddings --config configs/quantization_cassava.yaml
```

The t-SNE runner ([src/visualize_embeddings.py](src/visualize_embeddings.py))
hooks the input to the final classifier layer, projects the 1024-d features to
2D, and saves `outputs/student_quantized_cassava/tsne_embeddings.png`.

## Results

### Main study — Cassava, 5-fold cross-validation (mean ± std)

| Teacher | Student | Accuracy | Macro-F1 | Params |
|---|---|---|---|---|
| ResNet50 | *(supervised)* | 85.13 ± 0.95 | 74.00 ± 0.78 | 23,518,277 |
| EfficientNet-B2 | *(supervised)* | 84.59 ± 0.67 | 72.60 ± 0.61 | 7,708,039 |
| ViT-B/16 | *(supervised)* | 82.40 ± 0.64 | 69.38 ± 1.26 | 85,802,501 |
| — | MobileNetV3-Small (baseline) | 81.67 ± 1.01 | 68.22 ± 0.80 | 1,522,981 |
| ResNet50 | MobileNetV3-Small (distilled) | **83.49 ± 0.31** | 70.72 ± 1.99 | 1,522,981 |
| ViT-B/16 | MobileNetV3-Small (distilled) | 83.12 ± 0.72 | 69.85 ± 1.95 | 1,522,981 |
| EfficientNet-B2 | MobileNetV3-Small (distilled) | 83.04 ± 0.41 | 70.34 ± 1.26 | 1,522,981 |

All distilled students share the MobileNetV3-Small architecture; the teacher is
discarded after training. Macro-averaged precision and recall are also reported
in `outputs/cv_summary.{csv,md}`.

### Significance vs. the supervised baseline (paired t-test, accuracy)

| Teacher | Mean diff | p-value | Significant (p<0.05) |
|---|---|---|---|
| ResNet50 | +1.82 pp | 0.029 | yes |
| ViT-B/16 | +1.45 pp | 0.036 | yes |
| EfficientNet-B2 | +1.37 pp | 0.079 | no (borderline) |

### KD sensitivity ablation — ResNet50 teacher, single split (fold 0)

| Sweep | T | α | Accuracy | Macro-F1 |
|---|---|---|---|---|
| temperature | 1 | 0.7 | 83.50 | 71.35 |
| temperature | 2 | 0.7 | 83.79 | 71.29 |
| temperature | **4** | **0.7** | **84.14** | **71.52** |
| temperature | 8 | 0.7 | 83.67 | 71.37 |
| alpha | 4 | 0.3 | 82.78 | 70.61 |
| alpha | 4 | 0.5 | 83.60 | 70.89 |
| alpha | 4 | **0.7** | **84.14** | **71.52** |
| alpha | 4 | 0.9 | 83.11 | 69.39 |

The student is stable across both sweeps (within ~1 point), with a gentle peak at
the default **T=4.0, α=0.7** used in the main study — confirming the fixed choice
is well justified.

### Preliminary study — PlantVillage, 38 classes (single split, RTX 4070)

Students are mean ± SD over three training seeds; the teacher is a single run.

| Model | Strategy | Accuracy (%) | Macro-F1 (%) | Params |
|---|---|---|---|---|
| ResNet50 | Teacher supervised | 99.48 | 99.31 | 23,585,894 |
| MobileNetV3-Small | Supervised baseline | 97.84 ± 0.59 | 96.75 ± 0.73 | 1,556,806 |
| MobileNetV3-Small | Distilled from ResNet50 | **99.48 ± 0.05** | 99.29 ± 0.10 | 1,556,806 |

Paired KD gain **+1.64 ± 0.64 pts (p = 0.047)**. Note the standard deviations:
the distilled student is ~12× more stable across seeds than the supervised one,
so distillation removes run-to-run variance as well as raising accuracy. A single
run of this experiment is misleading — an earlier single run drew a lucky
baseline (98.51) and reported the gain as only +0.86.

### Edge compression — int8 PTQ on the Cassava student (it does not work)

Static rows are mean ± SD over **five calibration draws** — on this model a single
draw is not a measurement (the draw alone moves accuracy by ~7 points).

| Variant | Accuracy (%) | Macro-F1 (%) | Size (MB) |
|---|---|---|---|
| Float32 | 84.51 | 72.57 | 5.93 |
| Dynamic int8 *(classifier head only; 0/52 convs)* | 84.48 | 72.45 | 4.23 |
| Static int8, per-channel | 35.48 ± 7.63 | 20.25 | — |
| Static int8, per-channel + SE blocks in float32 | 35.97 ± 7.60 | 20.79 | — |
| Static int8, per-tensor | 10.10 ± 0.24 | 7.26 | — |

Quantizing the convolutions **collapses** the student (chance is 20%, the
majority class is 62%; per-tensor lands *below* chance, degenerating to a single
class). `src/quant_analysis.py` runs four controls, each ruling out a cause:

- **Not the pipeline, and not distillation.** ResNet50 quantizes losslessly
  through the identical path (86.51 → 86.65 ± 0.17), while all ten
  MobileNetV3-Small students — 5 seeds × {supervised, distilled} — collapse
  without exception (12.8–49.4%).
- **Not calibration-limited.** 256 / 1024 / 2048 calibration images give
  28.6 ± 6.0 / 35.5 ± 7.6 / 33.0 ± 5.8 — the spread *between* sizes is no larger
  than the draw noise *within* one size.
- **Not the SE blocks.** Holding them in float32 changes nothing
  (35.97 vs 35.48).
- **Not activation range.** ResNet50's peak activation is 535 vs the student's
  105 — the *wider*-range model is the one that survives.
- **The failure is distributed.** Quantizing one block-group at a time costs
  ≤0.22 pts; quantizing all of them costs 49 pts. No layer can be exempted.

Distillation does **not** improve quantizability (paired p = 0.52). QAT is a
prerequisite, not a refinement.

### Batch-one latency (idle machine; median of 5×300 timed passes)

| Model | Params | CPU (ms) | GPU (ms) |
|---|---|---|---|
| ResNet50 (teacher) | 23.5M | 100.3 | 4.68 |
| EfficientNet-B2 | 7.7M | 46.8 | 8.25 |
| ViT-B/16 | 85.8M | 129.9 | 7.34 |
| MobileNetV3-Small (student) | 1.5M | 15.5 | 3.96 |
| — + dynamic int8 | 1.5M | 17.2 | — |

**Parameter count is a poor proxy for latency.** EfficientNet-B2 has ⅓ of
ResNet50's parameters yet is ~2× slower on GPU. And the student's 15× parameter
reduction buys only **1.18× on GPU** but **6.5× on CPU** — at batch one a GPU is
launch-latency bound, so the case for a compact student is a case about CPU-class
hardware. Compression ratios and speed-ups are different currencies.

## Distillation loss

The student minimizes a weighted sum of hard-label cross-entropy and a
temperature-scaled KL divergence to the teacher's softened logits:

```
L = α · T² · KL(softmax(z_t / T) ‖ softmax(z_s / T)) + (1 − α) · CE(y, z_s)
```

with `T = 4.0` throughout, `α = 0.7` (Cassava) / `α = 0.5` (PlantVillage).

## Repository structure

```
knowledge-distillation-edge/
├── configs/
│   ├── cv_teacher_{resnet50,efficientnet_b2,vit_b16}_cassava.yaml   # teacher CV
│   ├── cv_baseline_student_cassava.yaml                             # student baseline CV
│   ├── cv_distillation_{resnet50,efficientnet_b2,vit_b16}_cassava.yaml  # distillation CV
│   ├── ablation_resnet50_cassava.yaml                              # KD sensitivity ablation
│   ├── quantization_cassava{,_mixed,_pertensor,_dynamic}.yaml      # int8 PTQ variants
│   ├── replicate_quant_cassava.yaml                                # 5 seeds x 2 arms, then quantize
│   ├── benchmark_latency_cassava.yaml                              # batch-one latency
│   ├── *_full_rtx4070.yaml                                         # PlantVillage preliminary
│   └── pv_replicate/                                               # PlantVillage seed replicates
├── src/
│   ├── cross_validate.py   # k-fold CV runner (baseline + distillation modes)
│   ├── collate_cv.py       # collate CV results + paired significance tests
│   ├── ablate_kd.py        # KD temperature/alpha sensitivity ablation
│   ├── quantize.py         # int8 PTQ (PT2E; static/mixed/per-tensor/dynamic)
│   ├── replicate_quant.py  # is the int8 collapse from distillation or the architecture?
│   ├── benchmark_latency.py# batch-one latency, with a built-in reliability check
│   ├── models/             # teacher/student model factory
│   ├── data/               # dataset loading, stratified k-fold splits
│   ├── evaluate.py         # metrics (accuracy, precision, recall, F1)
│   ├── train_baseline.py   # single-split supervised training
│   └── train_distillation.py  # single-split distillation training
├── outputs/                # per-run metrics, cv_summary.*, cv_significance.md
├── reports/report.tex      # full write-up
├── requirements.txt
└── README.md
```

## Reproducibility

All experiments are config-driven and seeded. Reported metrics are read from the
stored result files (`outputs/*/cv_results.json`, `cv_fold_metrics.csv`), not
copied from logs. The CV fold partition is fixed by seed, so every teacher and
the baseline see identical folds — a fair, paired comparison.
