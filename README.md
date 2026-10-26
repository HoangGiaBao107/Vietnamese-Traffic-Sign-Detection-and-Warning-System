# Vietnamese Traffic Sign Detection and Warning System

<p align="center">
  <strong>Traffic Sign Detection and Warning Model Applied on Edge Devices</strong><br>
  Vietnamese traffic sign detection using YOLO11n
</p>

<p align="center">
  <a href="https://vietnamese-traffic-sign-detection-and-warning-system-hgb.streamlit.app/">
    <strong>Live Web Application</strong>
  </a>
</p>

## Overview

This project develops a lightweight traffic sign detector and an audio warning interface for Vietnamese road environments. The main model is **YOLO11n trained with all 58 VR-TSD classes**. Its evaluation combines duplicate screening, a cluster-level train/validation/test split, offline augmentation of rare or historically difficult classes, and fixed-severity synthetic Fog and Rain stress tests.

The final training configuration and class coverage follow the finalized notebook. The numerical summaries below reproduce the accompanying science report: **mean ± standard deviation over three training seeds**, except where a single control run or a historical device measurement is explicitly identified. A result from one notebook execution should be distinguished from these three-run summaries.

## Dataset and Class Counts

The experiments use **VR-TSD (Vietnamese Real-World Traffic Sign Dataset)**.

| Property | Value |
|---|---:|
| Original traffic-sign classes | **58** |
| Classes retained in the main training configuration | **58** |
| Classes represented in the clean test split | **55** |
| Classes without a ground-truth instance in the test split | **3** |
| Original images | **8,078** |
| Original annotated bounding boxes | **13,016** |
| Image clusters | **6,813** |
| Training images before offline augmentation | **5,471** |
| Validation images | **1,341** |
| Test images | **1,266** |
| Additional offline-augmented training images | **1,970** |
| Rare/hard classes targeted by offline augmentation | **21** |

**58 is the number of classes in the main model; 55 is the number of classes that can be evaluated on this test split.** No class is removed from the main model. The notebook sets `USE_PRUNING = False`, and its saved output states `Classes used for training: 58`.

The three classes with no test instance remain in the training label space:

| Original class ID | Class name | Training instances | Validation instances | Test instances |
|---:|---|---:|---:|---:|
| 24 | `131b-cam-do-xe-ngay-le` | 2 | 0 | 0 |
| 43 | `301b-cac-xe-chi-duoc-re-phai` | 2 | 1 | 0 |
| 46 | `301e-cac-xe-chi-duoc-re-trai-2` | 2 | 0 | 0 |

Test-set class-averaged metrics cover the **55 classes with at least one ground-truth instance**. This also applies to the Fog/Rain derivatives of the test split. The report's separate **53-class variant is a pruning ablation**, not the main model or the number of evaluable test classes.

### Original Dataset Publication

Ngoc, D. N., Nhat, L. N. H., Van, Q. H., Nguyen, H., Bui, H., & Nguyen, P. A. (2026). *VR-TSD: A Real-World Dataset and a Lightweight YOLOv8n Model for Traffic Sign Recognition in Vietnam*. In *Proceedings of CSA 2025*, Lecture Notes in Electrical Engineering, vol. 1583, pp. 105–115. Springer, Singapore.

**DOI:** [10.1007/978-981-95-7896-2_10](https://doi.org/10.1007/978-981-95-7896-2_10)

**Publisher page:** [Springer Nature Link](https://link.springer.com/chapter/10.1007/978-981-95-7896-2_10)

The original VR-TSD publication provides the dataset source; the YOLO11n experiments reported here are this project's experiments.

## Data Preparation and Evaluation Protocol

1. Pool all **8,078 images** from the original train, validation, and test folders.
2. Link matching source filenames and near-duplicate images using a **256-bit DCT-based perceptual hash**, with a Hamming-distance threshold of **16**.
3. Merge linked images using **union-find clustering**, then assign whole clusters to train/validation/test, targeting **70/15/15**. The realized proportions are **67.7/16.6/15.7%**, because clusters are indivisible.
4. Verify **zero cross-split matches under the selected pHash threshold and zero shared source identifiers** for all three split pairs. This check does not exclude every possible residual dependence between frames from the same video.
5. Retain all **58 classes**. Apply bounding-box-aware offline augmentation only to training images containing a rare or historically difficult class.
6. Generate **four Fog levels and four Rain levels** from the held-out test split. Severity parameters are fixed before the reported evaluation; synthetic weather is not injected during training.
7. Select confidence thresholds on validation only. Evaluate all domains without test-time augmentation, using the same matching procedure for Precision, Recall, and F1.

Fog is procedural haze created through noise and alpha blending; it does not model depth-dependent atmospheric attenuation. Rain is a windshield-raindrop simulation using the external renderer. Both are synthetic stress tests rather than real adverse-weather driving footage.

## Model and Training

| Configuration | Value |
|---|---|
| Architecture | **YOLO11n**, initialized from COCO-pretrained weights |
| Main model label space | **58 classes** |
| Input image size | **896 × 896** |
| Training environment | **2 × NVIDIA T4** on Kaggle |
| Batch size, total | **64** |
| Maximum epochs | **200** |
| Optimizer | **AdamW** |
| Early-stopping patience | **35** |
| Initial learning rate | **0.0020** |
| Final learning-rate fraction | **0.01** |
| Learning-rate schedule | **Cosine** |
| Warm-up epochs | **8** |
| Fused parameter count | **2,593,462** |
| PyTorch weights | **5.6 MB** |
| ONNX export | **10.3 MB** |

Three confidence policies are compared: a fixed threshold of 0.25, one global threshold selected on validation, and per-class thresholds selected on validation. Classes with fewer than 15 validation instances retain 0.25 under the per-class policy.

## Results

### Clean Test Results

The main model has **58 classes**; the clean test set supports class-averaged evaluation on **55 classes**. Values are percentages, reported as **mean ± standard deviation over three seeds**. Threshold-dependent scores in this table use the **fixed confidence threshold of 0.25**, matching the report's Abstract and Conclusion.

| Metric | Result |
|---|---:|
| mAP@0.5 | **96.17 ± 0.32** |
| mAP@0.5:0.95 | **79.67 ± 0.70** |
| Macro Precision | **91.89 ± 0.66** |
| Macro Recall | **95.12 ± 0.05** |
| Macro F1 at IoU 0.5 | **93.11 ± 0.48** |
| Micro F1 at IoU 0.5 | **91.91 ± 0.28** |

### Validation, Clean Test, and Weather Stress Tests

The following table reproduces the report's domain comparison. Precision, Recall, and Macro F1 use the **validation-selected global threshold**, which is then kept fixed for the clean test and stress domains within each run. Consequently, their clean-test F1 differs from the fixed-threshold result above.

| Domain | mAP@0.5 | mAP@0.5:0.95 | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 94.20 ± 0.61 | 77.20 ± 0.36 | 87.24 ± 0.13 | 91.56 ± 1.56 | 87.24 ± 1.61 |
| Clean test | **96.17 ± 0.32** | **79.67 ± 0.70** | 90.54 ± 2.39 | 95.07 ± 0.65 | 92.34 ± 1.13 |
| Fog F1 | 95.95 ± 0.30 | 79.46 ± 0.32 | 89.57 ± 1.91 | 94.49 ± 0.14 | 91.49 ± 0.98 |
| Fog F2 | 95.28 ± 0.33 | 78.56 ± 0.40 | 88.92 ± 2.50 | 93.72 ± 0.36 | 90.79 ± 1.22 |
| Fog F3 | 93.11 ± 0.15 | 75.64 ± 0.27 | 87.06 ± 2.23 | 89.22 ± 2.04 | 87.40 ± 0.23 |
| Fog F4 | 80.16 ± 1.97 | 63.80 ± 1.52 | 80.66 ± 2.97 | 74.76 ± 4.26 | 75.84 ± 1.48 |
| Rain R1 | 94.89 ± 0.55 | 77.48 ± 0.80 | 90.04 ± 2.15 | 92.66 ± 0.90 | 90.92 ± 0.87 |
| Rain R2 | 91.39 ± 0.73 | 71.10 ± 0.50 | 86.98 ± 2.83 | 88.06 ± 1.74 | 86.94 ± 0.99 |
| Rain R3 | 87.10 ± 0.16 | 67.14 ± 0.48 | 84.18 ± 3.76 | 82.66 ± 1.43 | 82.27 ± 1.43 |
| Rain R4 | 84.57 ± 0.24 | 63.20 ± 0.34 | 83.22 ± 3.88 | 79.25 ± 2.30 | 80.35 ± 0.83 |

mAP@0.5 remains above 90% through **Fog F3** and **Rain R2**, with larger losses at the heavier severities.

### Confidence-Threshold Comparison

Clean test, percentages; mean ± standard deviation over three seeds.

| Threshold policy | Macro Precision | Macro Recall | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|
| Fixed, 0.25 | 91.89 ± 0.66 | 95.12 ± 0.05 | 93.11 ± 0.48 | 91.91 ± 0.28 |
| Global, selected on validation | 90.54 ± 2.39 | 95.07 ± 0.65 | 92.34 ± 1.13 | 91.27 ± 1.28 |
| Per-class, selected on validation | 94.28 ± 0.07 | 93.54 ± 0.28 | 93.54 ± 0.12 | 92.85 ± 0.09 |

### Offline Augmentation Ablation

The full pipeline is summarized over three seeds; the no-offline-augmentation control is a **single run, seed 2102**. The difference is **without augmentation minus with augmentation**, in percentage points.

| Metric | With offline augmentation | Without offline augmentation | Difference |
|---|---:|---:|---:|
| Clean test mAP@0.5 | 96.17 ± 0.32 | 95.34 | −0.83 |
| Clean test mAP@0.5:0.95 | 79.67 ± 0.70 | 78.10 | −1.57 |
| Validation mAP@0.5 | 94.20 ± 0.61 | 92.00 | −2.20 |
| Clean test Macro F1, fixed 0.25 | 93.11 ± 0.48 | 92.58 | −0.53 |
| Fog F3 mAP@0.5 | 93.11 ± 0.15 | 89.51 | −3.60 |
| Fog F4 mAP@0.5 | 80.16 ± 1.97 | 74.45 | −5.71 |
| Rain R3 mAP@0.5 | 87.10 ± 0.16 | 84.08 | −3.02 |
| Rain R4 mAP@0.5 | 84.57 ± 0.24 | 78.53 | −6.04 |

### Historical Inference Benchmarks

These are the throughput values reported in the science report. As stated in its limitations, the on-device measurements used an **earlier 53-class YOLO11n export**. They are historical deployment measurements, not a new benchmark of the final 58-class export.

| Platform | ms / image | FPS |
|---|---:|---:|
| NVIDIA T4 ×2, Kaggle | 5.78 | 173.01 |
| Google Pixel 5 | 58.0 | 17.24 |
| Samsung Galaxy S23 | 12.7 | 78.74 |

Inference throughput differs from the end-to-end speed of the Streamlit application, which also performs image/video loading, drawing, display, and audio handling.

## Web Application

The Streamlit interface provides:

- **Image Analysis:** upload a JPG, JPEG, or PNG image.
- **Video Analysis:** upload an MP4, AVI, or MOV video.
- Detection overlays and corresponding audio warnings.

<p align="center">
  <img src="web_interface.png" alt="Streamlit web interface" width="95%">
</p>

**Live application:** [Open the Streamlit demo](https://vietnamese-traffic-sign-detection-and-warning-system-hgb.streamlit.app/)

## Run Locally

```bash
git clone https://github.com/HoangGiaBao107/Vietnamese-Traffic-Sign-Detection-and-Warning-System.git
cd Vietnamese-Traffic-Sign-Detection-and-Warning-System/web_deployment
pip install -r requirements.txt
streamlit run app.py
```

The deployed ONNX model and the class-to-subtitle/audio mappings must use the same class order. Use the final 58-class export when deploying the final 58-class research model.

## Repository Files

| Path | Purpose |
|---|---|
| `CODE PIPELINE.ipynb` | Training and evaluation notebook |
| `raindrops.py` | Windshield-raindrop renderer |
| `Science report.pdf` | Science report |
| `model results/` | Model export and result figures |
| `web_deployment/app.py` | Streamlit application entry point |
| `web_deployment/config.py` | Model path, class subtitles, audio paths, and application settings |
| `web_deployment/detector.py` | Inference, visual overlays, and warning triggers |
| `web_deployment/utils.py` | Application utilities |
| `web_deployment/best.onnx` | Model export used by the application |
| `web_deployment/Warning_sound/` | Audio warning files |
| `web_deployment/requirements.txt` | Python dependencies for the application |

## Limitations and Next Steps

- Only **55 of 58 classes** have test instances, and 14 of those classes have fewer than ten test instances.
- The duplicate-control result is bounded by the selected hash threshold and source identifiers; complete independence of frames from the same source video is not established.
- Some classes remain unevenly distributed across splits.
- Fog and Rain are synthetic; real adverse-weather driving footage and in-vehicle validation remain necessary.
- The report uses three training seeds and one no-offline-augmentation control run. The effects of class balancing and the photometric/blur augmentations have not been separated.
- The final 58-class mobile export still requires re-benchmarking. Audio-warning latency and false-alarm behavior have not been evaluated.

## Author

**Hoang Gia Bao**  
Phan Boi Chau High School for the Gifted, Nghe An, Vietnam  
Email: [hgbao0907@gmail.com](mailto:hgbao0907@gmail.com)

## Links

- [GitHub repository](https://github.com/HoangGiaBao107/Vietnamese-Traffic-Sign-Detection-and-Warning-System)
- [Live web application](https://vietnamese-traffic-sign-detection-and-warning-system-hgb.streamlit.app/)
- [Original VR-TSD publication](https://doi.org/10.1007/978-981-95-7896-2_10)
