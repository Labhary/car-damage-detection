# Car Damage Detection with Visual Secret Sharing

A deep learning project that classifies car images as **damaged** or **not damaged** using a CNN binary classifier. The project also integrates a **visual secret sharing** scheme — a cryptographic technique that splits a car image into multiple shares that reveal nothing individually, but reconstruct the original when combined.

---

## What it does

**Part 1 — Damage Detection:**
- Takes a car image as input
- Classifies it as damaged or not damaged using a trained CNN
- Displays the result with confidence

**Part 2 — Visual Secret Sharing:**
- Splits the classified car image into N shares using XOR-based secret sharing
- Each share looks like random noise — reveals nothing alone
- Reconstructs the original image by XOR-ing all shares together
- Both parts are wired into a Tkinter GUI

---

## How it works

### 1. Dataset
- Images stored in `dataa/` folder, organized into subfolders by class (damaged / not damaged)
- Input size: **256×256×3** (RGB)
- Corrupted or unsupported image formats are automatically removed before training

### 2. Preprocessing
- Images are loaded using `tf.keras.utils.image_dataset_from_directory`
- Pixel values normalized to `[0, 1]`
- Split: **70% train / 20% validation / 10% test**

### 3. CNN Model Architecture

Binary classification — output is a single sigmoid neuron (0 = damaged, 1 = not damaged).

```
Input (256×256×3)
        │
   Conv2D(16, 3×3, relu) → MaxPooling2D
        │
   Conv2D(32, 3×3, relu) → MaxPooling2D
        │
   Conv2D(16, 3×3, relu) → MaxPooling2D
        │
   Flatten
        │
   Dense(256, relu)
        │
   Dense(1, sigmoid)
```

- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Epochs**: 10
- **Metrics**: Accuracy, Precision, Recall

### 4. Visual Secret Sharing (XOR-based)

The image is split into N random shares:
- Share 1 through N-1 are randomly generated
- Share N = XOR of all previous shares with the original image
- Reconstruction: XOR all N shares together → original image restored perfectly

This guarantees that any subset of shares (fewer than N) gives zero information about the original image.

### 5. GUI
A Tkinter interface lets you:
- Load a car image
- Run damage detection and see the result with classification label
- Generate shares of the image and save them
- Load shares and reconstruct the original image

---

## Project structure

```
CarImageClassification/
├── CarStateClassifier.ipynb     # Full pipeline: training, evaluation, GUI
├── dataa/                       # Training images (organized by class)
│   ├── damaged/
│   └── notdamaged/
├── models/
│   └── imageclassifier.h5       # Saved trained model
├── logs/                        # TensorBoard training logs
├── folder_001/ ... folder_007/  # Generated image shares
├── damaged.jpg                  # Sample damaged car image
├── notdamaged.jpg               # Sample not-damaged car image
└── .gitignore
```

---

## Requirements

```
Python 3.11
tensorflow
opencv-python (cv2)
numpy
matplotlib
Pillow
```

Install all dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib pillow
```

---

## Usage

### Train the model
Open `CarStateClassifier.ipynb` in Jupyter and run cells 1 through 11 (up to "Save the Model"). The trained model will be saved under `models/imageclassifier.h5`.

### Run damage detection on a single image
```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('models/imageclassifier.h5')
img = cv2.imread('your_car.jpg')
resize = tf.image.resize(img, (256, 256))
yhat = model.predict(np.expand_dims(resize / 255, 0))

if yhat > 0.5:
    print('Car is not damaged')
else:
    print('Car is damaged')
```

### Run the full GUI
Run the GUI cells in the notebook (cells 51 and 53). A window will open for loading images, detecting damage, generating shares, and reconstructing.

---

## Notes

- GPU memory growth is configured automatically to avoid OOM errors during training
- TensorBoard logs are saved in `logs/` — run `tensorboard --logdir logs` to visualize training
- The visual secret sharing uses XOR which is lossless — reconstruction is pixel-perfect
- The model expects **256×256 RGB** images; other sizes are resized automatically
