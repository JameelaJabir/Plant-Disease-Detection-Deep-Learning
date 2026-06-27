# Plant Disease Detection Using Deep Learning

### Comparative Study of CNN, VGG16, Deep CNN & MobileNetV2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red)
![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-green)
![Classes](https://img.shields.io/badge/Classes-38-yellow)
![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-96.46%25-brightgreen)

---

## Overview

Plant diseases are a major threat to global food security, causing significant crop yield loss every year. Early and accurate detection of these diseases is critical for timely treatment and prevention.

This project builds and compares four deep learning models to automatically classify **38 plant disease categories** from leaf images using the **PlantVillage dataset**. The study evaluates lightweight baselines, custom deep architectures, and transfer learning approaches.

**Models Compared:**

| # | Model | Approach | Accuracy |
|---|-------|----------|----------|
| 1 | Simple CNN | Custom baseline | 90.85% |
| 2 | VGG16 | Transfer learning (frozen) | 87.09% |
| 3 | Deep CNN | Custom deep architecture | 95.41% |
| 4 | MobileNetV2 | Transfer learning + fine-tuning | **96.46%** |

---

## Dataset

**PlantVillage Dataset** - [Kaggle Link](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

| Property | Value |
|----------|-------|
| Total Images | 54,305 |
| Classes | 38 (healthy + diseased) |
| Training Set | 43,444 images (80%) |
| Validation/Test Set | 10,861 images (20%) |
| Input Resolution | 124 × 124 pixels |
| Batch Size | 16 |
| Split Seed | 123 (reproducible) |

**Crops included:** Apple, Tomato, Cotton, Corn, Pepper, Potato, Soybean, Raspberry, Blueberry, Cherry, Grape, Orange, Peach, Squash, Strawberry, and more.

---

## Preprocessing & Data Augmentation

All models share the same preprocessing pipeline:

- Resize images to **124 × 124 pixels**
- Normalize pixel values: `1/255` rescaling
- Apply data augmentation (training only):
  - Random Horizontal Flip
  - Random Vertical Flip
  - Random Rotation (factor: 0.2)
  - Random Zoom (factor: 0.1)

---

## Model Architectures

---

### 1. Simple CNN - 90.85%

Lightweight baseline model with three convolutional blocks.

```
Input (124 × 124 × 3)
│
├── Rescaling (1/255)
│
├── Conv2D (16 filters, 3×3) → ReLU
│   └── MaxPooling2D (2×2)
│
├── Conv2D (32 filters, 3×3) → ReLU
│   └── MaxPooling2D (2×2)
│
├── Conv2D (64 filters, 3×3) → ReLU
│   └── MaxPooling2D (2×2)
│
├── Flatten
├── Dense (128) → ReLU
└── Dense (38) → Softmax
```

**Role:** Establishes a simple but solid baseline at 90.85% accuracy without any pre-trained weights.

---

### 2. VGG16 - 87.09% *(Focus Model)*

> This is the primary model explored in this study.

VGG16 is a deep convolutional neural network developed by the Visual Geometry Group at Oxford, pre-trained on **ImageNet** (1.2 million images, 1000 classes). It is well known for its simplicity — a deep stack of 3×3 convolutions — and its strong feature representations.

#### Architecture Used

```
Input (124 × 124 × 3)
│
├── Rescaling (1/255)
├── Data Augmentation (flip, rotation, zoom)
│
├── VGG16 Base [FROZEN — ImageNet pretrained]
│   │
│   ├── Block 1: Conv2D (64) × 2 → MaxPool
│   ├── Block 2: Conv2D (128) × 2 → MaxPool
│   ├── Block 3: Conv2D (256) × 3 → MaxPool
│   ├── Block 4: Conv2D (512) × 3 → MaxPool
│   └── Block 5: Conv2D (512) × 3 → MaxPool
│       (Total: 13 convolutional layers)
│
├── Flatten
├── Dense (256) → ReLU
├── Dropout (0.5)
└── Dense (38) → Softmax  ← custom classifier head
```

#### Parameter Breakdown

| Layer Group | Parameters | Size |
|-------------|-----------|------|
| VGG16 Base (frozen) | 14,714,688 | 56.13 MB |
| Custom Classifier Head | 1,189,670 | 4.54 MB |
| **Total** | **15,904,358** | **60.67 MB** |

Only the **classifier head (1.19M params)** is trained. The VGG16 base weights remain frozen throughout training.

#### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | sparse_categorical_crossentropy |
| Epochs | 15 |
| Batch Size | 16 |
| Early Stopping | patience = 5, monitor = val_loss |
| Model Checkpoint | saves best val_accuracy |

#### Training History (Epoch-by-Epoch)

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|-------|--------------|-------------|---------|
| 1 | 42.47% | 73.98% | - |
| 5 | ~72% | ~83% | - |
| 7 | 79.99% | 85.40% | - |
| 10 | ~81% | ~86% | - |
| 15 | 83.24% | **87.09%** | 0.3944 |

The model converged steadily but showed a noticeable **train-validation gap** throughout — train accuracy lagged behind validation accuracy in early epochs, suggesting the frozen VGG16 features were already rich, but the single linear head struggled to fully leverage them.

#### Why VGG16 Underperformed

Although VGG16 is a powerful backbone, several factors explain the 87.09% result:

1. **Frozen base weights** - ImageNet features are not perfectly aligned with plant disease textures. Fine-tuning the upper conv layers would have improved domain adaptation.
2. **Input size mismatch** - VGG16 was originally designed for 224×224 inputs. Using 124×124 means the spatial feature maps are smaller, potentially losing fine-grained texture details critical for disease classification.
3. **No fine-tuning phase** - Unlike MobileNetV2, VGG16 was not given a second training phase where deeper layers are unfrozen and allowed to adapt to the domain.
4. **Large model, small dataset relative to capacity** - VGG16 has 14.7M frozen parameters encoding ImageNet priors that may not generalize well to macro leaf patterns without domain tuning.

#### VGG16 vs MobileNetV2 - Key Difference

| Aspect | VGG16 | MobileNetV2 |
|--------|-------|-------------|
| Base frozen | Yes (all layers) | Phase 1 only |
| Fine-tuning | None | Last 30 layers unfrozen |
| Architecture efficiency | Heavy (60 MB) | Lightweight (~14 MB) |
| Feature alignment | ImageNet only | ImageNet + domain-tuned |
| Final accuracy | 87.09% | 96.46% |

The ~9% accuracy gap is almost entirely explained by fine-tuning. VGG16 with the same two-phase strategy would likely reach competitive performance.

---

### 3. Deep CNN - 95.41%

A custom deep architecture with 4 convolutional blocks and batch normalization.

```
Input (124 × 124 × 3)
│
├── Block 1: Conv2D (32) + BN → ReLU, Conv2D (32) + BN → ReLU → MaxPool
├── Block 2: Conv2D (64) + BN → ReLU, Conv2D (64) + BN → ReLU → MaxPool
├── Block 3: Conv2D (128) + BN → ReLU, Conv2D (128) + BN → ReLU → MaxPool
├── Block 4: Conv2D (256) + BN → ReLU, Conv2D (256) + BN → ReLU → MaxPool
│
├── Flatten
├── Dense (512) → ReLU
├── Dropout (0.5)
└── Dense (38) → Softmax
```

**Key advantage:** BatchNormalization after every conv layer stabilizes training and allows the model to learn richer representations despite training from scratch. Progressive filter doubling (32 → 64 → 128 → 256) builds increasingly abstract features.

---

### 4. MobileNetV2 - 96.46% (Best)

```
Input (124 × 124 × 3)
│
├── Data Augmentation
├── Rescaling (1/255)
│
├── MobileNetV2 Base [Depthwise Separable Convolutions]
│   └── Inverted Residual Blocks (efficient feature extraction)
│
├── GlobalAveragePooling2D
├── Dense (128) → ReLU
├── Dropout (0.3)
└── Dense (38) → Softmax
```

**Two-phase training:**
- **Phase 1:** MobileNetV2 base frozen → train only the top classifier
- **Phase 2:** Unfreeze last 30 layers → fine-tune with low learning rate

This domain adaptation strategy is the main reason MobileNetV2 outperforms all other models.

---

## Results Summary

| Model | Test Accuracy | Trainable Params | Training Strategy |
|-------|--------------|-----------------|-------------------|
| Simple CNN | 90.85% | All layers | From scratch |
| VGG16 | 87.09% | Top head only | Frozen base |
| Deep CNN | 95.41% | All layers | From scratch |
| **MobileNetV2** | **96.46%** | Top + last 30 layers | Transfer + fine-tune |

---

## Key Findings

- **MobileNetV2** achieves the best accuracy (96.46%) through its efficient depthwise separable convolutions and two-phase fine-tuning strategy.
- **VGG16** underperforms relative to its reputation because its base was fully frozen with no fine-tuning and input size did not match its original training resolution.
- **Deep CNN** (95.41%) proves that a well-designed custom architecture with BatchNorm can outperform a frozen transfer learning model, even without pre-trained weights.
- **Simple CNN** (90.85%) is a strong baseline given its minimal complexity — showing the task is learnable even with shallow feature extractors.
- Data augmentation (flip, rotation, zoom) was crucial in improving generalization across all four models.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/Plant-Disease-Detection-Deep-Learning.git
cd Plant-Disease-Detection-Deep-Learning
```

**2. Install dependencies**
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

**3. Download the dataset**

Download from [Kaggle PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and place it in the project directory.

**4. Run notebooks**
```bash
jupyter notebook
```

- `Dl_PlantVillage.ipynb` - Simple CNN, Deep CNN, MobileNetV2
- `Dl_PlantVillage_VGG16.ipynb` - VGG16 transfer learning model

---

## Technologies

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| TensorFlow / Keras | Model building and training |
| NumPy / Pandas | Data handling |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Evaluation metrics |
| Jupyter / Google Colab | Development environment |

---

## Contributors

| Student ID | Name |
|-----------|------|
| IT22076052 | Mahindarathna D.G.P.T.D |
| IT22060662 | Jameela M.J.F |
| IT22011398 | Kariyawasam M.G.P.Y |
| IT22032324 | Gunawickrama I.U.S.U.G |

---

## License

This project is developed for academic and research purposes.
