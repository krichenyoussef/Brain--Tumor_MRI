# Brain Tumor MRI: Classification and Tumor Localization

![CI](https://github.com/krichenyoussef/Brain--Tumor_MRI/actions/workflows/ci.yml/badge.svg)

Classifies brain MRI slices into **glioma, meningioma, pituitary tumor or no tumor**, with a focus on what
matters in medical imaging: a test set free of duplicate images, the number of **missed tumors**, calibrated
confidence, and **Grad-CAM** heatmaps that show where the model looked.

Three transfer-learning backbones are compared on the same clean split: **ResNet-50**, **EfficientNet-B0**
and a small **Vision Transformer (ViT-S/16)**.

The [`detection/`](detection/) folder adds **tumor localization**: a bounding box around the tumor.
The dataset has no box labels, so they were built from Grad-CAM heatmaps, checked by hand on 240 images,
and used to train YOLO detectors. A benchmark compares four localization methods and four classifier backbones.

> Research and education only. This is not a medical device and must not be used for diagnosis.

## Results

### Tumor localization

Scored against 48 human-reviewed test boxes plus 20 no-tumor images. A box counts as correct when it overlaps
the reviewed box by at least 50% and has the right class (mAP@0.5).

| Method | mAP@0.5 | 95% interval | Tumor images boxed correctly | ms / image |
|---|---:|---:|---:|---:|
| Grad-CAM, raw | 0.03 | 0.00 to 0.10 | 10% | 29 |
| Grad-CAM + brightness refinement | 0.40 | 0.26 to 0.59 | 54% | 38 |
| YOLO11n trained on 5,559 automatic boxes | 0.54 | 0.39 to 0.69 | 60% | 15 |
| YOLO11n trained on 199 reviewed boxes | **0.68** | 0.55 to 0.81 | **69%** | **11** |

![Detection mAP@0.5 by method](detection/outputs/bench_detection.png)

A small set of clean boxes beat a large set of noisy ones. The test set is small, so the intervals overlap.
Full results and caveats: [detection/BENCHMARK.md](detection/BENCHMARK.md).

### Classification on the original Kaggle split

Four backbones trained with identical settings, 8 epochs, tested on the 1,600 Kaggle Testing images.
This split may contain duplicates of training images, so treat these numbers as an upper bound.

| Backbone | Params (M) | Accuracy | Macro F1 | Glioma recall | Images / s |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 27.8 | **95.8%** | **0.957** | **84.2%** | 402 |
| EfficientNet-B0 | 4.0 | 95.1% | 0.950 | 82.0% | 1,015 |
| ResNet-50 | 23.5 | 95.0% | 0.949 | 81.2% | 498 |
| MobileNetV3-Large | 4.2 | 94.8% | 0.946 | 81.0% | **1,774** |

Every backbone misses 16 to 19% of gliomas while the other classes stay near 99 to 100%. The problem comes from
the data, not the architecture.

### Classification on the clean, duplicate-free split

Run the pipeline (below), then paste the table from `results/RESULTS.md` here.

| Model | Params (M) | Accuracy | Macro F1 | Tumor recall | Missed tumors | ECE |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | | | | | | |
| EfficientNet-B0 | | | | | | |
| ViT-S/16 | | | | | | |

*Tumor recall* = share of tumor scans not classified as "no tumor". *ECE* = expected calibration error
(lower means the model's confidence matches its accuracy).

<!-- After training, add: results/<model>/confusion_matrix.png, reliability.png and gradcam_examples.png -->

## Why duplicate removal matters

Public brain MRI datasets often contain the same scan more than once: exact copies, re-encoded copies, or
images placed in both the training and testing folders. If a scan appears in both train and test, the model
is graded on images it has already seen, and accuracy looks better than it is.

`src/dedup.py` hashes every image (MD5 for exact copies, perceptual hash for re-encoded ones), merges
duplicates into groups, drops groups whose copies carry different labels, then builds a stratified
70 / 15 / 15 train / validation / test split by group. Nothing can leak across splits. The numbers it finds
are written to `results/dedup_report.json`.

## Pipeline

```mermaid
flowchart LR
    A[Kaggle images<br/>Training/ and Testing/] --> B[Deduplicate<br/>MD5 + perceptual hash]
    B --> C[Clean split<br/>train / val / test]
    C --> D[Fine-tune<br/>ResNet-50 · EfficientNet-B0 · ViT-S]
    D --> E[Best epoch on validation]
    E --> F[Test metrics<br/>recall · missed tumors · ECE]
    E --> G[Grad-CAM]
    F --> H[RESULTS.md]
    G --> I[Gradio demo]
```

- **Training:** AdamW, cosine learning-rate schedule, light augmentation (crop, flip, small rotation,
  brightness/contrast), mixed precision on GPU, early stopping on validation macro F1.
- **Model selection** uses the validation split only. The test split is touched once, at the end.

## Quickstart

### On Google Colab (recommended, free GPU)
Open `notebooks/train_colab.ipynb` in Colab, select a GPU runtime and run all cells. It downloads the data,
removes duplicates, trains the three models, and produces the table and figures.

### Locally
```bash
pip install -r requirements-dev.txt

# 1. Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
#    Extract it so that you have data/raw/Training/<class>/ and data/raw/Testing/<class>/
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/raw --unzip

# 2. Remove duplicates and build the split
python -m src.dedup

# 3. Train and evaluate (one model, or all three)
python -m src.train --model resnet50 --epochs 15
python -m src.train --model all

# 4. Comparison table and Grad-CAM examples
python -m src.compare
python -m src.gradcam --checkpoint checkpoints/resnet50.pt
```

## Demo

```bash
cp checkpoints/<best_model>.pt checkpoints/best.pt
python app/app.py
```

Upload a scan to get the class probabilities and the Grad-CAM heatmap.

To publish it on **Hugging Face Spaces** (free): create a Gradio Space, then upload `app/app.py` as
`app.py`, `app/requirements.txt` as `requirements.txt`, the `src/` folder, and your checkpoint as
`checkpoints/best.pt`.

## Tumor localization (`detection/`)

The `detection/` folder goes one step further than classification: it draws a **bounding box around the tumor**.
The Kaggle dataset has no box labels, so they were built in three stages:

1. **Automatic boxes for all 5,400 tumor images,** from Grad-CAM heatmaps refined by tumor brightness.
2. **Manual review of 240 images,** where each box was accepted, redrawn, or dropped when no tumor was visible.
   About 60% of the automatic boxes matched the reviewed box.
3. **YOLO-format datasets** for training a detector.

The results are in the [Results](#results) section above.
See [detection/README.md](detection/README.md) for every step and [detection/BENCHMARK.md](detection/BENCHMARK.md)
for the full benchmark and its caveats.
The detection scripts read the images from the same `data/raw/` folder as this project.

```bash
cd detection
pip install -r requirements.txt
python src/train_classifier.py                       # about 10 min on a laptop GPU, saves outputs/model.pt
python src/predict.py ../data/raw/Testing/glioma     # classify a folder of scans
```

## Project structure

```
src/
├── config.py      # classes, backbones, paths
├── dedup.py       # duplicate detection and leak-free split
├── data.py        # dataset and transforms
├── models.py      # timm backbones and Grad-CAM target layers
├── train.py       # training, early stopping, test evaluation
├── metrics.py     # tumor recall, missed tumors, calibration, plots
├── gradcam.py     # heatmaps for one image or a grid of test images
└── compare.py     # results table across models
app/app.py         # Gradio demo
notebooks/         # Colab training notebook
tests/             # pytest suite on synthetic images (runs in CI, no dataset needed)
detection/         # bounding-box labels, YOLO datasets and benchmark (own README and requirements)
```

## Tests

```bash
python -m pytest -q
```

The tests build a small synthetic dataset with planted duplicates (an exact train/test leak, a re-encoded
copy, a scan with two labels) and check that the deduplication catches all of them. They also run the full
pipeline end to end: train, evaluate, compare and Grad-CAM.

## Limitations

- Trained on 2D slices from a single public dataset; performance on scans from other hospitals or machines
  is unknown.
- Labels come from the dataset and are not clinically verified.
- Grad-CAM shows where the model looked, not why it is right; a plausible heatmap does not prove a
  correct diagnosis.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and evaluation details.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
by Masoud Nickparvar on Kaggle. Check its license on the dataset page before redistributing images.
The images are not stored in this repository.

## License

Code under the MIT License.
