| Method | mAP@0.5 | 95% interval | mAP@0.5:0.95 | Box right place, any class (AP@0.5) | Tumor images boxed correctly | False alarms on no-tumor | ms / image |
|---|---|---|---|---|---|---|---|
| Grad-CAM v1 | 0.027 | 0.00 to 0.10 | 0.010 | 0.012 | 10% | 0% | 29 |
| Grad-CAM v2 (refined) | 0.402 | 0.26 to 0.59 | 0.330 | 0.371 | 54% | 0% | 38 |
| YOLO11n, 199 reviewed imgs | 0.676 | 0.55 to 0.81 | 0.245 | 0.677 | 69% | 0% | 11 |
| YOLO11n, 5,559 auto imgs | 0.536 | 0.39 to 0.69 | 0.316 | 0.540 | 60% | 0% | 15 |

| Method | Glioma AP@0.5 | Meningioma AP@0.5 | Pituitary AP@0.5 |
|---|---|---|---|
| Grad-CAM v1 | 0.042 | 0.039 | 0.000 |
| Grad-CAM v2 (refined) | 0.413 | 0.565 | 0.226 |
| YOLO11n, 199 reviewed imgs | 0.582 | 0.896 | 0.549 |
| YOLO11n, 5,559 auto imgs | 0.547 | 0.597 | 0.465 |

| Backbone | Params | Test acc | Macro F1 | Glioma | Meningioma | No tumor | Pituitary | Train time | Latency bs1 | Throughput bs32 |
|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny | 27.82M | 95.8% | 0.957 | 84.2% | 99.2% | 100.0% | 99.8% | 6 min | 5.8 ms | 402 img/s |
| efficientnet_b0 | 4.01M | 95.1% | 0.950 | 82.0% | 98.5% | 100.0% | 100.0% | 3 min | 6.5 ms | 1015 img/s |
| resnet50 | 23.52M | 95.0% | 0.949 | 81.2% | 99.0% | 100.0% | 99.8% | 5 min | 5.0 ms | 498 img/s |
| mobilenet_v3_large | 4.21M | 94.8% | 0.946 | 81.0% | 98.5% | 99.8% | 99.8% | 2 min | 5.0 ms | 1774 img/s |
