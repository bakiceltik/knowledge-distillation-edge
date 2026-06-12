| Experiment | Teacher | Student | Params | Accuracy % | Precision % | Recall % | Macro-F1 % | Weighted-F1 % |
|---|---|---|---|---|---|---|---|---|
| cv_teacher_resnet50_cassava | - | resnet50 | 23,518,277 | 85.13 ± 0.95 | 75.08 ± 1.40 | 73.69 ± 1.28 | 74.00 ± 0.78 | 85.11 ± 0.85 |
| cv_teacher_efficientnet_b2_cassava | - | efficientnet_b2 | 7,708,039 | 84.59 ± 0.67 | 73.91 ± 1.51 | 71.75 ± 0.86 | 72.60 ± 0.61 | 84.44 ± 0.51 |
| cv_distill_resnet50_cassava | resnet50 | mobilenet_v3_small | 1,522,981 | 83.49 ± 0.31 | 72.70 ± 0.66 | 69.74 ± 2.62 | 70.72 ± 1.99 | 83.17 ± 0.62 |
| cv_distill_vit_b16_cassava | vit_b_16 | mobilenet_v3_small | 1,522,981 | 83.12 ± 0.72 | 72.90 ± 0.99 | 68.25 ± 2.36 | 69.85 ± 1.95 | 82.64 ± 0.91 |
| cv_distill_efficientnet_b2_cassava | efficientnet_b2 | mobilenet_v3_small | 1,522,981 | 83.04 ± 0.41 | 71.75 ± 1.53 | 69.40 ± 1.98 | 70.34 ± 1.26 | 82.76 ± 0.47 |
| cv_teacher_vit_b16_cassava | - | vit_b_16 | 85,802,501 | 82.40 ± 0.64 | 71.41 ± 1.28 | 69.08 ± 1.58 | 69.38 ± 1.26 | 82.14 ± 0.64 |
| cv_mobilenetv3_baseline_cassava | - | mobilenet_v3_small | 1,522,981 | 81.67 ± 1.01 | 69.83 ± 2.64 | 67.57 ± 1.38 | 68.22 ± 0.80 | 81.46 ± 0.70 |

_Cross-validated results (5-fold), mean ± standard deviation across folds. Precision, recall, and F1 are macro-averaged._
