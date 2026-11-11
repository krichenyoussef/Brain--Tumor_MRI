#!/usr/bin/env bash
# Benchmark chain: detectors, classifier backbones, then detection scoring.
cd "/c/Users/Youss/OneDrive/Bureau/Brain_Tumor"
L=outputs/bench_log.txt; : > $L
echo "[$(date +%T)] yolo gold start" >> $L
python src/train_yolo.py gold 100 > outputs/yolo_gold_stdout.txt 2>&1; echo "[$(date +%T)] yolo gold exit $?" >> $L
python src/train_yolo.py full 40 > outputs/yolo_full_stdout.txt 2>&1; echo "[$(date +%T)] yolo full exit $?" >> $L
for a in resnet50 efficientnet_b0 convnext_tiny mobilenet_v3_large; do
  ARCH=$a OUT_DIR=outputs/cls_$a python -u src/train_classifier.py > outputs/cls_${a}_stdout.txt 2>&1
  echo "[$(date +%T)] cls $a exit $?" >> $L
done
(cd src && python bench_detection.py) > outputs/bench_detection_stdout.txt 2>&1; echo "[$(date +%T)] bench exit $?" >> $L
echo DONE >> $L
