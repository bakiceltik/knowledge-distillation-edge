# Project Plan

## Objective

Study whether knowledge distillation can transfer the performance of a stronger image classifier to a lightweight MobileNetV3 student for edge-oriented plant disease classification on PlantVillage.

## Dataset

- Dataset: PlantVillage
- Task: multi-class image classification of plant diseases
- Planned handling: standardized train, validation, and test splits with reproducible preprocessing

## Baselines

- Supervised teacher baseline using `ResNet50`
- Supervised student baseline using `MobileNetV3`
- Future optional teacher comparison with EfficientNet variants if time permits

## Proposed Method

Train a teacher model first, then optimize a MobileNetV3 student with a hybrid loss that combines standard cross-entropy and teacher-guided distillation loss from softened logits. The later phase of the project may also explore quantization to improve deployment efficiency on edge hardware.

## Planned Ablations

- Temperature sensitivity experiments for distillation
- `alpha` sensitivity experiments balancing hard-label and soft-target losses
- Comparison between supervised student training and distilled student training

## Planned Error Analysis

- Inspect confusion patterns across disease classes
- Review classes with frequent misclassification
- Visualize representative failure cases
- Compare teacher and student mistakes to identify where knowledge transfer helps or fails

## Expected Risks

- Class imbalance or split quality may affect reliable evaluation
- Teacher quality may limit distillation gains
- Limited project time may constrain full quantization experiments
- Deployment-oriented latency measurements may depend on available hardware
