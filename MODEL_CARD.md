# Model card: Brain Tumor MRI Classifier

## Model
Image classifiers fine-tuned from ImageNet-pretrained backbones (via `timm`): ResNet-50, EfficientNet-B0 and
ViT-S/16. Input: one 2D brain MRI slice, resized to 224 × 224. Output: probabilities for four classes
(glioma, meningioma, pituitary tumor, no tumor) and a Grad-CAM heatmap.

## Intended use
Education, research and portfolio demonstration of a careful medical-imaging workflow.

**Out of scope:** any clinical, diagnostic or triage use. The model has not been validated by clinicians,
on external data, or under any regulatory framework.

## Data
Kaggle "Brain Tumor MRI Dataset" (Masoud Nickparvar). Before training, exact and near-duplicate images are
grouped, groups with conflicting labels are removed, and one image per group is split 70 / 15 / 15 into
train / validation / test, stratified by class. The duplicate statistics are in
`results/dedup_report.json`.

## Evaluation
Reported on the held-out test split, using the epoch with the best validation macro F1:

- accuracy and macro F1;
- **tumor recall** and **number of missed tumors** (tumor scans predicted as "no tumor"), the most costly
  error;
- per-class precision, recall and F1, and the confusion matrix;
- **expected calibration error** and a reliability diagram.

Results: see `results/RESULTS.md`.

## Limitations and risks
- Single public source; no external validation. Expect lower performance on other scanners, protocols or
  populations.
- 2D slices only; real diagnosis uses full 3D volumes, several sequences and clinical context.
- Dataset labels are not clinically re-verified.
- Grad-CAM heatmaps can look convincing even when the prediction is wrong.
