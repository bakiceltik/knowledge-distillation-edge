# Paired significance vs. supervised baseline (accuracy)

Paired two-sided t-test on per-fold values (folds shared across runs).

| Teacher | Folds | Mean diff (pp) | t | p | Significant (p<0.05) |
|---|---|---|---|---|---|
| resnet50 | 5 | +1.82 | 3.321 | 0.0294 | yes |
| vit_b_16 | 5 | +1.45 | 3.098 | 0.0363 | yes |
| efficientnet_b2 | 5 | +1.37 | 2.347 | 0.0787 | no |
