<div align="center">

# 🧠 Brain Tumor Detection System

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/>

> **Intelligent Brain Tumor Detection from MRI Scans**  
> A Machine Learning + Deep Learning application for classifying 4 types of brain tumors with high accuracy.

<br/>

```
  ██████╗ ██████╗  █████╗ ██╗███╗   ██╗    ███╗   ███╗██████╗ ██╗
  ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║    ████╗ ████║██╔══██╗██║
  ██████╔╝██████╔╝███████║██║██╔██╗ ██║    ██╔████╔██║██████╔╝██║
  ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║    ██║╚██╔╝██║██╔══██╗██║
  ██████╔╝██║  ██║██║  ██║██║██║ ╚████║    ██║ ╚═╝ ██║██║  ██║██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝
              D E T E C T I O N   S Y S T E M
```

</div>

---

## 📌 Overview

The **Brain Tumor Detection** system is a web application that combines Deep Learning and Machine Learning to analyze brain MRI images, assisting doctors and researchers in diagnosing brain tumors quickly and accurately.

### ✨ Key Highlights

| Feature | Description |
|---------|-------------|
| 🔬 **Dual-Model Inference** | Simultaneous comparison of KNN and SVM predictions |
| 🧬 **ResNet18 Feature Extraction** | Deep feature extraction from MRI scans |
| 📊 **PCA Dimensionality Reduction** | Optimized feature space compression |
| 🖼️ **Visual Bounding Box** | Visual annotation of suspected tumor regions |
| 📈 **Confidence Score** | Detailed probability report per class |
| 🌐 **Web Interface** | User-friendly Flask-based interface |

---

## 🎯 Tumor Classes

```
┌─────────────────┬──────────────────┬─────────────────┬──────────────────┐
│    🔴 Glioma    │  🟠 Meningioma   │  ✅ No Tumor    │  🟣 Pituitary   │
├─────────────────┼──────────────────┼─────────────────┼──────────────────┤
│ Glial cell tumor│ Meninges tumor,  │ Normal brain    │ Pituitary gland  │
│ high malignancy │ usually benign,  │ scan, no tumor  │ tumor, affects   │
│ rate            │ slow-growing     │ detected        │ hormone control  │
└─────────────────┴──────────────────┴─────────────────┴──────────────────┘
```

---

## 🏗️ System Architecture

```
                        📤 Upload MRI Image
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Flask Web Server  │
                    │     (app.py)        │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Image Preprocessing│
                    │  Resize(224x224)    │
                    │  Normalize + ToTensor│
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   ResNet18          │
                    │  Feature Extractor  │◄── ImageNet Pretrained
                    │  (512-dim vector)   │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   PCA Reduction     │
                    │  pca_model.pkl      │
                    └────────┬────────────┘
                             │
               ┌─────────────┴──────────────┐
               ▼                            ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │     KNN Model       │    │     SVM Model       │
    │   knn_model.pkl     │    │   svm_model.pkl     │
    └─────────┬───────────┘    └─────────┬───────────┘
              │                          │
              ▼                          ▼
    ┌─────────────────────────────────────────────────┐
    │          Result Visualization & Report          │
    │     Bounding Box + Label + Confidence Score     │
    └─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
brain_detection/
│
├── 📂 src/
│   ├── 🐍 dataset.py          # Dataset loading and preprocessing pipeline
│   ├── 🧠 model.py            # ResNet50-based BrainTumorClassifier architecture
│   └── 🏋️ train.py            # Model training and evaluation scripts
│
├── 📂 model/
│   ├── 🤖 knn_model.pkl       # Trained K-Nearest Neighbors model
│   ├── 🤖 svm_model.pkl       # Trained Support Vector Machine model
│   └── 📉 pca_model.pkl       # PCA dimensionality reduction model
│
├── 📂 data/
│   ├── 📂 train/              # Training MRI images
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── 📂 test/               # Testing MRI images
│
├── 📂 Brain-Tumor-Test-Images/ # Sample images for quick testing
│
├── 📂 templates/
│   ├── DiseaseDet.html         # Home page
│   ├── uimg.html              # Image upload page
│   ├── pred.html              # Prediction result page
│   └── error.html             # Error page
│
├── 📂 static/                 # CSS, JavaScript, static assets
├── 📂 upload/                 # Uploaded image storage
├── 🐍 app.py                  # Main Flask web application
├── 🧪 test.py                 # Model testing script
└── 📋 requirements.txt        # Python package dependencies
```

---

## ⚙️ Installation & Setup

### 📋 System Requirements

- Python **3.8+**
- CUDA **11.8+** *(recommended for GPU acceleration)* or CPU
- Minimum **8GB** RAM

### 🚀 Step-by-Step Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/<your-username>/brain_detection.git
cd brain_detection
```

**Step 2: Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**Step 3: Install dependencies**

> ⚠️ **With GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask scikit-learn pillow numpy joblib
```

> 💻 **CPU only:**
```bash
pip install -r requirements.txt
```

**Step 4: Launch the application**
```bash
python app.py
```

**Step 5: Open in your browser**
```
http://localhost:5000
```

---

## 🧪 Training the Model

To retrain from scratch:

```bash
# Prepare the dataset
python src/dataset.py

# Train the model
python src/train.py
```

> 📌 Recommended dataset: [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
> Organize `data/train/` and `data/test/` directories with sub-folders per class.

---

## 📊 Image Processing Pipeline

```python
# Step 1: Image preprocessing
transform = Compose([
    Resize((224, 224)),       # Standardize dimensions
    ToTensor(),                # Convert to tensor
    Normalize(                 # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Step 2: Feature extraction via ResNet18
# Output: 512-dimensional feature vector

# Step 3: PCA dimensionality reduction
# Boost speed & reduce noise

# Step 4: Parallel classification
# ├── KNN → class label + confidence
# └── SVM → class label + confidence
```

---

## 🖥️ Web Interface

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page with system overview |
| `/uimg` | GET | MRI image upload form |
| `/uimg` | POST | Process image & return classification results |

### 📤 Supported Image Formats

```
✅ PNG    ✅ JPG / JPEG    ✅ GIF
```
> 📦 Maximum file size: **16 MB**

---

## 🔬 Technical Details

### Feature Extractor — ResNet18

```python
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()   # Remove final classification head
resnet.eval()               # Set to inference mode
```

### BrainTumorClassifier — ResNet50 (Fine-tuned)

```python
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=2):
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Fine-tune all layers
        self.resnet.fc = nn.Linear(n_inputs, num_classes)
```

### Dual Classification Pipeline

| Model | Strengths | Limitations |
|-------|-----------|-------------|
| **KNN** | Simple, interpretable, no training phase | Slow on large datasets |
| **SVM** | Powerful in high-dimensional spaces | Requires careful hyperparameter tuning |

---

## 📈 Results & Output

> Results are displayed directly on the web interface, including:
- 🖼️ Original uploaded MRI image
- 🔲 KNN-annotated image with bounding box and label
- 🔲 SVM-annotated image with bounding box and label
- 📊 Class probabilities for all 4 categories: Glioma, Meningioma, No Tumor, Pituitary

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. 🍴 Fork this repository
2. 🌿 Create your feature branch: `git checkout -b feature/your-feature-name`
3. 💾 Commit your changes: `git commit -m 'feat: Add feature X'`
4. 📤 Push to the branch: `git push origin feature/your-feature-name`
5. 📬 Open a Pull Request

---

## 👨‍💻 Author

<div align="center">

| | Information |
|---|---|
| 👤 **Full Name** | Huynh Van An |
| 🎓 **Student ID** | 2200011724 |
| 🏫 **University** | *(add your university name)* |
| 📧 **Email** | *(add your email)* |
| 🔗 **GitHub** | *(add your GitHub profile)* |

</div>

---

## 📄 License

This project is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 📚 References

- [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet: Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn: KNN & SVM](https://scikit-learn.org/stable/)

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

<img src="https://img.shields.io/github/stars/yourusername/brain_detection?style=social"/>

*Made with ❤️ and 🧠 by Huynh Van An*

</div>
