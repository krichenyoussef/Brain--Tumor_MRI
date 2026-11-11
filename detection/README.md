# Brain tumor MRI: classification and bounding-box labels

The Kaggle dataset has 7,200 MRI slices in 4 classes (glioma, meningioma, notumor, pituitary) with **class labels only**.
This project turns it into a **detection dataset** (boxes around the tumor) without a radiologist, in 7 steps,
then benchmarks detectors and classifiers on it.

This folder is the detection part of the repository. The repository root holds the main classification project,
which builds a duplicate-free split. **Run every command below from inside this `detection/` folder.**

> These boxes were drawn by a non-expert (an AI reviewer) and by an automatic method. They are fine for learning and
> prototyping object detection. They are **not** clinical ground truth.

## Folder layout

| Path | What it is |
|---|---|
| `../data/raw/` | Original images, shared with the main project and never modified (see Setup) |
| `src/` | All scripts, one per step. `src/paths.py` finds the images |
| `outputs/` | Model, reports, review images |
| `labels/` | All box files (CSV + YOLO txt) |
| `yolo_full/`, `yolo_gold/` | Ready-to-train YOLO datasets |
| `BENCHMARK.md` | Benchmark results and charts |

## Step 1: Environment

Python 3.12, PyTorch 2.9 + CUDA, an RTX 3050 Ti (4 GB), plus OpenCV, scikit-learn, and ultralytics. Nothing extra needed installing.

## Step 2: Classifier (`src/train_classifier.py`)

- ResNet50 pretrained on ImageNet, final layer replaced by 4 outputs.
- 224×224 input. Light augmentation: horizontal flip, ±10° rotation, brightness/contrast jitter.
- 90% of `Training` for training and 10% for validation (stratified). `Testing` is used once at the end.
- AdamW, lr 3e-4, one-cycle schedule, label smoothing 0.05, mixed precision, 8 epochs (~10 min on the GPU).

Results (`outputs/test_report.txt`, `outputs/confusion_matrix.png`):

| | Validation | Test |
|---|---|---|
| Accuracy | 99.6% | 95.6% |

Almost all test errors are gliomas predicted as meningioma or no tumor (glioma recall 83.75%). Validation loss stopped improving at
epoch 6, so more epochs would not help. The gap comes from Testing glioma images that look different from the Training ones.

## Step 3: Automatic boxes, version 1 (`src/gradcam_boxes.py`)

1. Run Grad-CAM on the last ResNet block for the **true** class. This gives a 7×7 heat map of where the model sees evidence.
2. Upsample it to the image size and mask out everything outside the head.
3. Threshold at 45% of the maximum and keep the blob with the most heat. Its rectangle is the box.

Output: `labels/auto_boxes.csv` and `labels/auto/`. Problem: a 7×7 map is coarse, so boxes were often 2–3× too big or shifted.

## Step 4: Automatic boxes, version 2 (`src/refine_boxes.py`)

Most tumors here are bright on contrast-enhanced T1 MRI, so inside the Grad-CAM region the script:

1. Erodes the head mask to drop the bright skull ring.
2. Keeps pixels above the 93rd brightness percentile of the brain.
3. Picks the bright blob with the most Grad-CAM support and a sensible size. Its rectangle, plus 8% padding, is the box.
4. Falls back to the version-1 box when no blob qualifies.

Output: `labels/auto_boxes_v2.csv` (the `method` column says `bright` or `cam`).

## Step 5: Manual review (`src/annot_sheet.py`, `src/manual_to_corrections.py`)

- Random sample (seed 0) of **240 images**: 60 Training + 20 Testing per tumor class (`labels/review_list_*.csv`).
- Each image was rendered at 480 px with a 10% grid (`outputs/annot/<class>_NNN.png`).
- For each image the reviewer chose one verdict:
  - **ok**: the auto box is good.
  - **fix**: a new box, in percent coordinates.
  - **drop**: no tumor clearly visible on this slice.
- Results are in `labels/manual_review.csv`, then converted to pixel boxes in `labels/corrections.csv` and `labels/reviewed_ok.csv`.

| Class | ok | fix | drop |
|---|---|---|---|
| Glioma | 22 | 26 | 32 |
| Meningioma | 45 | 30 | 5 |
| Pituitary | 30 | 34 | 16 |

## Step 6: How good are the automatic boxes? (`src/eval_auto_vs_manual.py`)

IoU (overlap) of each auto box with the reviewed box, on the 187 reviewed images that kept a box:

| Class | v1 mean IoU | v1 IoU ≥ 0.5 | v2 mean IoU | v2 IoU ≥ 0.5 |
|---|---|---|---|---|
| Glioma | 0.29 | 12.5% | 0.59 | 47.9% |
| Meningioma | 0.34 | 14.7% | 0.74 | 70.7% |
| Pituitary | 0.10 | 0.0% | 0.56 | 54.7% |
| All | 0.25 | 9.1% | 0.64 | 59.4% |

The v2 numbers are slightly optimistic because an "ok" verdict counts as a perfect match.
Typical failures:
- **Pituitary:** the box locks onto bright sphenoid or clivus fat, or falls back to a large box on the brainstem.
- **Meningioma:** the box covers only the brightest part of a large mass.
- **Glioma:** many slices show no clear lesion at all.

Spot-check sheets: `outputs/annot/gold_check_*.png`. Cyan is the reviewed box, green is the v2 auto box.

## Step 7: Final labels and YOLO export (`src/apply_corrections.py`, `src/build_yolo_dataset.py`)

`labels/final_boxes.csv` has a `status` column for every image:

| status | count | meaning |
|---|---|---|
| auto | 5,160 | v2 automatic box, not reviewed |
| verified | 97 | reviewed, auto box accepted |
| corrected | 90 | reviewed, box redrawn |
| dropped | 53 | reviewed, no visible tumor, excluded |
| background | 1,800 | no-tumor images, empty label |

YOLO class IDs: 0 = glioma, 1 = meningioma, 2 = pituitary.

| Dataset | Train | Val |
|---|---|---|
| `yolo_full` | 5,559 images (4,159 tumor boxes, mostly automatic) + no-tumor backgrounds | 68 reviewed Testing images (48 boxes + 20 backgrounds) |
| `yolo_gold` | 199 images (139 reviewed boxes + 60 backgrounds) | the same 68 |

Both use the same reviewed validation set, so detector scores are measured against trusted boxes only.
That set is small (48 boxes), so treat its mAP as a rough estimate.

## Rerun everything

```bash
python src/train_classifier.py
python src/gradcam_boxes.py
python src/refine_boxes.py
python src/manual_to_corrections.py
python src/apply_corrections.py
python src/eval_auto_vs_manual.py
python src/build_yolo_dataset.py full
python src/build_yolo_dataset.py gold
```

## Step 8: Benchmark

Two benchmarks compare methods on the same test data. Results and charts are in [BENCHMARK.md](BENCHMARK.md).

**Detection.** Four ways of boxing the tumor are scored against the human-reviewed Testing boxes:
48 tumor images plus 20 no-tumor images (`src/bench_detection.py`).

| Method | How it works |
|---|---|
| Grad-CAM v1 | Classifier predicts the class, step-3 box for that class, no box if it predicts no tumor |
| Grad-CAM v2 | Same, with the step-4 brightness refinement |
| YOLO11n gold | Detector trained on `yolo_gold`, the 199 reviewed images (`src/train_yolo.py gold`) |
| YOLO11n full | Detector trained on `yolo_full`, the 5,559 mostly automatic images (`src/train_yolo.py full`) |

The Grad-CAM methods must predict the class themselves, like a real detector, rather than use the true class.
Metrics are mAP@0.5, mAP@0.5:0.95, class-agnostic AP@0.5, the share of tumor images boxed correctly,
the false-alarm rate on no-tumor images, a 95% bootstrap interval, and speed per image.
YOLO is scored with `last.pt`, not `best.pt`, because `best.pt` is picked on this same test set.

**Classification.** Four ImageNet-pretrained backbones are trained with identical settings:
ResNet50, EfficientNet-B0, ConvNeXt-Tiny and MobileNetV3-Large.
They are compared on test accuracy, macro F1, per-class recall, size, training time and inference speed.
Select a backbone with the `ARCH` environment variable and an output folder with `OUT_DIR`.

**Caveats.**
- The test set is small, with 48 boxes, so the bootstrap intervals are wide.
- Some reviewed test boxes are Grad-CAM v2 boxes accepted as correct. This favors Grad-CAM v2, and to a lesser degree
  YOLO full, which was trained on v2-style boxes.

## Rerun the benchmark

```bash
python src/train_yolo.py gold 100
python src/train_yolo.py full 40
ARCH=efficientnet_b0 OUT_DIR=outputs/cls_efficientnet_b0 python src/train_classifier.py
cd src && python bench_detection.py && cd ..
python src/make_benchmark_charts.py
```

`outputs/run_bench.sh` runs the whole chain, including all four backbones.

## Setup

```bash
pip install -r requirements.txt
```

The dataset is not in the repository. Download the Kaggle "Brain Tumor MRI Dataset" once, at the repository root,
exactly as the main project's README describes: `data/raw/Training/<class>/` and `data/raw/Testing/<class>/`.
Both projects then use the same copy. `src/paths.py` looks in `../data/raw`, then `../data`, then `data/`,
and the `BRAIN_TUMOR_DATA` environment variable overrides all of them.

Model weights, the YOLO dataset copies and the review sheets are excluded to keep the repository small.
The scripts above regenerate them. `yolo_full/` and `yolo_gold/` must be rebuilt on a new machine,
because their `data.yaml` files store an absolute path.
