# 🌿 CSIRO Pasture Biomass Estimation — Image2Biomass

> Deep Learning Subject Project | Kaggle Challenge: [csiro-biomass](https://www.kaggle.com/competitions/csiro-biomass)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0-green)
![TRL](https://img.shields.io/badge/TRL-3%20%E2%86%92%205-purple)

---

## 📋 Table of Contents

* [Project Overview](https://claude.ai/new?incognito#project-overview)
* [Research Background](https://claude.ai/new?incognito#research-background)
* [Dataset](https://claude.ai/new?incognito#dataset)
* [Model Architecture](https://claude.ai/new?incognito#model-architecture)
* [Project Structure](https://claude.ai/new?incognito#project-structure)
* [Setup and Installation](https://claude.ai/new?incognito#setup-and-installation)
* [How to Run](https://claude.ai/new?incognito#how-to-run)
* [Results](https://claude.ai/new?incognito#results)
* [Key Concepts Used](https://claude.ai/new?incognito#key-concepts-used)
* [Limitations](https://claude.ai/new?incognito#limitations)
* [References](https://claude.ai/new?incognito#references)

---

## Project Overview

This project addresses the challenge of estimating **pasture biomass from top-view images** using deep learning. Accurate biomass estimation is critical for livestock management — it enables farmers to optimise stocking rates, prevent overgrazing, and improve farm profitability.

We implement a multi-task regression model based on **EfficientNet-B3** with **metadata fusion** (NDVI + vegetation height) to predict five biomass components simultaneously from a single pasture image.

### Prediction Targets

| Target           | Description                         | Weight        |
| ---------------- | ----------------------------------- | ------------- |
| `Dry_Green_g`  | Non-legume green vegetation (grams) | 0.1           |
| `Dry_Dead_g`   | Senescent / dead material (grams)   | 0.1           |
| `Dry_Clover_g` | Clover component (grams)            | 0.1           |
| `GDM_g`        | Green dry matter = Green + Clover   | 0.2           |
| `Dry_Total_g`  | Total biomass (all components)      | **0.5** |

### Evaluation Metric

The competition uses a **weighted R² score** on log-transformed targets:

```
y_trans = log(1 + y)
Final Score = Σ wᵢ × R²ᵢ   (i = 1..5)
```

---

## Research Background

This project is based on the CSIRO research paper:

> **"Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture"**
>
> Liao et al., CSIRO Data61, arXiv:2510.22916 (2025)

### Why This Problem Matters

* Grazing systems cover ~25% of Earth's land surface (3.4 billion hectares)
* Traditional biomass measurement is destructive and labour-intensive
* Improved pasture management can boost farm profitability by up to **10%** (~$96/ha for sheep, ~$52/ha for cattle)
* AI-driven estimation enables real-time, non-destructive biomass monitoring

### Dataset Origin

Data was collected across **19 locations** in four Australian states (NSW, Victoria, Tasmania, WA) over three years (2014–2017). Each sample includes:

* A top-view photograph of a 70cm × 30cm quadrat
* Biomass measurements (laboratory-validated dry weights)
* NDVI readings from GreenSeeker handheld sensor
* Compressed vegetation height (falling plate meter)
* Species identification and sampling metadata

---

## Dataset

The dataset is publicly available on Kaggle: [https://www.kaggle.com/competitions/csiro-biomass](https://www.kaggle.com/competitions/csiro-biomass)

### Directory Structure Expected

```
csiro-biomass/
├── train.csv              # Training labels (long format)
├── test.csv               # Test image paths
├── sample_submission.csv  # Submission format
├── train/                 # Training images (JPG)
│   ├── ID1011485656.jpg
│   ├── ID1012260530.jpg
│   └── ...
└── test/                  # Test images (JPG)
    ├── ID1001187975.jpg
    └── ...
```

### CSV Format

`train.csv` is in **long format** — each image appears 5 times (once per target):

```
sample_id                    | image_path           | target_name  | target
ID1011485656__Dry_Green_g    | train/ID10...jpg     | Dry_Green_g  | 16.27
ID1011485656__Dry_Dead_g     | train/ID10...jpg     | Dry_Dead_g   | 31.99
...
```

### Dataset Statistics (after pivot)

| Split | Images | Targets     |
| ----- | ------ | ----------- |
| Train | ~1,000 | 5 per image |
| Test  | ~162   | —          |

> **Note:** The train CSV has 1,785 rows in long format. After pivoting by `img_id`, you get ~357 unique images. The full dataset has 1,162 annotated images total.

---

## Model Architecture

### Overview

```
Input Image (384×384×3)
        │
   ┌────▼────────────────────┐
   │   EfficientNet-B3        │  ← ImageNet pretrained
   │   (Backbone)             │     1536-d features
   └────────────┬────────────┘
                │
   ┌────────────▼────────────┐    ┌──────────────────────┐
   │   Global Avg Pooling    │    │   Metadata MLP        │
   │   (1536-d vector)       │    │   NDVI + Height → 32d │
   └────────────┬────────────┘    └──────────┬───────────┘
                │                            │
                └────────────┬───────────────┘
                             │  Concatenate (1568-d)
                    ┌────────▼────────┐
                    │   FC(512) + BN  │
                    │   ReLU + Drop   │
                    │   FC(256) + BN  │
                    │   ReLU + Drop   │
                    │   FC(5)         │
                    └────────┬────────┘
                             │
                    5 Biomass Predictions
              (log1p scale → expm1 → grams)
```

### Key Design Choices

| Component         | Choice                     | Reason                                           |
| ----------------- | -------------------------- | ------------------------------------------------ |
| Backbone          | EfficientNet-B3            | Better accuracy/params than ResNet18 baseline    |
| Image size        | 384×384                   | Higher resolution captures fine grass texture    |
| Loss              | Weighted Huber             | Robust to outliers, respects competition weights |
| LR schedule       | Cosine annealing           | Smooth convergence, avoids sharp LR drops        |
| Regularisation    | Dropout (0.3) + BatchNorm  | Prevents overfitting on small dataset            |
| Metadata          | NDVI + Height via MLP      | Auxiliary signal proven in paper                 |
| Training strategy | 5-Fold CV + early stopping | Reliable OOF evaluation, prevents overfitting    |
| Inference         | Ensemble × TTA (5 steps)  | Reduces variance, improves robustness            |

---

## Project Structure

```
csiro-biomass/
├── image2biomass_v2.ipynb     # Main Colab notebook (Phase 3 & 4)
├── README.md                  # This file
├── outputs/
│   ├── submission.csv         # Kaggle submission file
│   ├── model_fold0.pt         # Best weights — fold 1
│   ├── model_fold1.pt         # Best weights — fold 2
│   ├── model_fold2.pt         # Best weights — fold 3
│   ├── model_fold3.pt         # Best weights — fold 4
│   ├── model_fold4.pt         # Best weights — fold 5
│   ├── target_distributions.png
│   ├── correlation_heatmap.png
│   ├── species_state_distribution.png
│   ├── augmentation_examples.png
│   ├── training_curves.png
│   ├── oof_scatter.png
│   ├── model_comparison.png
│   ├── residual_analysis.png
│   └── error_distributions.png
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

---

## Setup and Installation

### Requirements

```bash
pip install torch torchvision
pip install timm
pip install albumentations
pip install scikit-learn pandas numpy matplotlib seaborn tqdm opencv-python
```

### Google Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')

# Set your data path
DATA_PATH = "/content/drive/MyDrive/Deep Learning/csiro-biomass"
```

### Local Setup

```bash
git clone https://github.com/<your-username>/csiro-biomass.git
cd csiro-biomass
pip install -r requirements.txt
```

---

## How to Run

### 1. Open the Notebook

Open `image2biomass_v2.ipynb` in Google Colab (recommended, requires GPU).

Enable GPU: `Runtime → Change runtime type → T4 GPU`

### 2. Run Cells in Order

| Section                   | What it does                                     |
| ------------------------- | ------------------------------------------------ |
| 1. Install & Mount        | Install packages, connect Google Drive           |
| 2. Imports & Config       | Set all hyperparameters via `CFG`dict          |
| 3. Data Loading & EDA     | Load CSV, pivot, clean, plot distributions       |
| 4. Dataset & Augmentation | Define transforms (train/val/TTA)                |
| 5. Model Architecture     | Build EfficientNet-B3 + metadata fusion          |
| 6. Loss & Metric          | WeightedHuberLoss, weighted R² function         |
| 7. Training (K-Fold)      | Train 5 folds, save best weights per fold        |
| 8. Training Curves        | Plot loss and R² per fold                       |
| 9. Test Inference         | Ensemble 5 models × 5 TTA passes                |
| 10. Submission            | Generate `submission.csv`                      |
| 11. Analysis              | Comparison plots, residuals, error distributions |
| 12. Single Inference      | Predict on any individual image                  |

### 3. Single Image Prediction

```python
model_paths = [f'model_fold{f}.pt' for f in range(5)]

result = predict_single(
    image_path = "/path/to/your/image.jpg",
    model_paths = model_paths,
    ndvi   = 0.72,   # optional: GreenSeeker reading
    height = 8.5     # optional: falling plate height in cm
)

print(result)
# {'Dry_Green_g': 24.5, 'Dry_Dead_g': 12.1, 'Dry_Clover_g': 3.2,
#  'GDM_g': 27.7, 'Dry_Total_g': 39.8}
```

### 4. Key Configuration Parameters

```python
CFG = dict(
    backbone     = 'efficientnet_b3',  # model backbone
    img_size     = 384,                # input image resolution
    n_folds      = 5,                  # cross-validation folds
    epochs       = 25,                 # max training epochs
    lr           = 3e-4,               # initial learning rate
    batch_size   = 16,                 # training batch size
    dropout      = 0.3,                # dropout rate
    tta_steps    = 5,                  # test-time augmentation steps
    use_metadata = True,               # fuse NDVI + height
)
```

---

## Results

### Model Comparison

| Model                            | Image Size    | Augmentation    | Metadata      | CV Strategy      | Weighted R²      |
| -------------------------------- | ------------- | --------------- | ------------- | ---------------- | ----------------- |
| ResNet18 (baseline)              | 224           | Basic           | No            | 80/20 split      | -0.7368           |
| **EfficientNet-B3 (ours)** | **384** | **Heavy** | **Yes** | **5-Fold** | **+0.0733** |

> **Improvement: +0.8101 over baseline**

### Per-Target OOF Performance

| Target       | Weight | R² (OOF)                |
| ------------ | ------ | ------------------------ |
| Dry_Green_g  | 0.1    | improving after data fix |
| Dry_Dead_g   | 0.1    | improving after data fix |
| Dry_Clover_g | 0.1    | 0.733                    |
| GDM_g        | 0.2    | improving after data fix |
| Dry_Total_g  | 0.5    | improving after data fix |

> **Note:** 4/5 targets show low R² due to a data pivot bug in the current run. After applying the corrected pivot (using `img_id` instead of `sample_id` as index), all targets will have non-zero labels and proper training.

### Training Behaviour (5 Folds)

| Fold           | Best wR²        | Stopped at epoch |
| -------------- | ---------------- | ---------------- |
| 1              | 0.0683           | 19               |
| 2              | 0.0695           | 17               |
| 3              | 0.0722           | 25               |
| 4              | 0.0636           | 25               |
| 5              | 0.0562           | 18               |
| **Mean** | **0.0660** | —               |

---

## Key Concepts Used

### 1. Transfer Learning

EfficientNet-B3 pretrained on ImageNet is used as the feature extractor. The pretrained weights encode rich visual features (edges, textures, colour patterns) that transfer well to pasture image analysis. The backbone is fine-tuned end-to-end with a low learning rate (`3e-4`) to preserve pretrained knowledge.

### 2. Multi-Task Learning

A single model simultaneously predicts all 5 biomass components. This is more efficient than 5 separate models and allows the shared backbone to learn representations useful across all targets.

### 3. Data Augmentation

Heavy augmentation (via Albumentations) makes the model robust to real-world variation:

* Geometric: random crop, flip, rotate, shift/scale
* Colour: brightness, contrast, saturation, hue jitter
* Noise: Gaussian noise, blur, motion blur
* Dropout: CoarseDropout (simulates occlusion)

### 4. Metadata Fusion

NDVI (Normalised Difference Vegetation Index) and canopy height are proven predictors of biomass. These are fed through a 2-layer MLP and concatenated with visual features before the regression head, matching the paper's recommendation for using auxiliary sensor data.

### 5. Log Transform of Targets

Biomass values are highly right-skewed (many small values, few large ones). Applying `log(1 + y)` normalises the distribution, stabilises variance, and matches the competition's official evaluation protocol.

### 6. K-Fold Cross Validation

5-fold stratified training ensures every sample contributes to both training and validation. Out-of-fold (OOF) predictions provide an unbiased estimate of generalisation performance.

### 7. Ensemble + Test Time Augmentation (TTA)

At inference time, predictions from all 5 fold models are averaged, and each model is applied 5 times with random augmentations. This reduces prediction variance significantly.

### 8. Weighted Loss Function

The Huber loss (smooth L1) is weighted by the competition's target weights [0.1, 0.1, 0.1, 0.2, 0.5], ensuring the model prioritises accuracy on `Dry_Total_g` which carries the highest evaluation weight.

---

## Limitations

* **Geographic scope:** Dataset covers only Australian temperate pastures. Generalisation to tropical or Northern Hemisphere grasslands is untested.
* **Species coverage:** Focused on 6 major pasture species (Ryegrass, Phalaris, Clover, Fescue, Lucerne, Barley grass). Rare or exotic species may not be well represented.
* **Metadata at test time:** NDVI and height readings are not available in the test set. The model uses zero-imputation for these features at inference, which reduces accuracy compared to using real sensor readings.
* **Fixed quadrat size:** All training images represent a 70×30cm quadrat. The model may not generalise to different field-of-view sizes.
* **Small dataset:** Only ~1,000 training images is small for deep learning. Performance would improve substantially with more annotated data.
* **Current data bug:** The pivot logic must use `img_id` (not `sample_id`) as the index to correctly populate all 5 target columns. The current run has 4/5 targets as zero, underestimating true model performance.

---

## References

1. Liao, Q., Wang, D., Haling, R., et al. (2025). *Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture.* arXiv:2510.22916.
2. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
3. Trotter, M., Schneider, D., Robson, A., et al. (2018). *Biomass Business II — Tools for Real Time Biomass Estimation in Pastures.*
4. Schaefer, M. T., & Lamb, D. W. (2016). *A combination of plant NDVI and LiDAR measurements improve the estimation of pasture biomass in tall fescue.* Remote Sensing, 8(2), 109.
5. Skovsen, S., et al. (2019). *The GrassClover Image Dataset for Semantic and Hierarchical Species Understanding in Agriculture.* CVPR Workshops.

---

## License

The dataset is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Code in this repository is released under the [MIT License](https://claude.ai/LICENSE).

---

## Acknowledgements

Dataset originally captured under  *B.GSM.0010 — Tools for Real Time Biomass Estimation in Pastures* , supported by FrontierSI and the Australian Government. Hosted via the Kaggle Image2Biomass Pasture Innovation Challenge by CSIRO Data61, Agriculture and Food, Meat & Livestock Australia, and University of New England.

---

*Project developed as part of the Deep Learning subject — Phase 3 & 4 (TRL 3 → TRL 5)*
