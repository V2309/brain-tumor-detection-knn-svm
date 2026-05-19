<div align="center">

# 🧠 Brain Tumor Detection System

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/>

> **Hệ thống phát hiện khối u não thông minh từ ảnh MRI**  
> Ứng dụng Machine Learning kết hợp Deep Learning để phân loại 4 loại u não với độ chính xác cao.

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

## 📌 Tổng Quan

Hệ thống **Brain Tumor Detection** là một ứng dụng web tích hợp Deep Learning và Machine Learning để phân tích ảnh MRI não, hỗ trợ bác sĩ và nhà nghiên cứu trong việc chẩn đoán khối u não nhanh chóng và chính xác.

### ✨ Điểm Nổi Bật

| Tính năng | Mô tả |
|-----------|-------|
| 🔬 **Dual-Model Inference** | So sánh đồng thời kết quả từ KNN và SVM |
| 🧬 **ResNet18 Feature Extraction** | Trích xuất đặc trưng sâu từ ảnh MRI |
| 📊 **PCA Dimensionality Reduction** | Tối ưu hóa không gian đặc trưng |
| 🖼️ **Visual Bounding Box** | Hiển thị vùng nghi ngờ trực quan trên ảnh |
| 📈 **Confidence Score** | Báo cáo xác suất chi tiết theo từng lớp |
| 🌐 **Web Interface** | Giao diện Flask dễ sử dụng |

---

## 🎯 Các Loại U Não Được Phân Loại

```
┌─────────────────┬──────────────────┬─────────────────┬──────────────────┐
│    🔴 Glioma    │  🟠 Meningioma   │  ✅ No Tumor    │  🟣 Pituitary   │
├─────────────────┼──────────────────┼─────────────────┼──────────────────┤
│  U tế bào thần  │  U màng não, phổ │  Não bình thường│  U tuyến yên,   │
│  kinh đệm, nguy │  biến, thường    │  không phát hiện│  ảnh hưởng đến  │
│  hiểm cao       │  lành tính       │  khối u         │  nội tiết       │
└─────────────────┴──────────────────┴─────────────────┴──────────────────┘
```

---

## 🏗️ Kiến Trúc Hệ Thống

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

## 📁 Cấu Trúc Dự Án

```
brain_detection/
│
├── 📂 src/
│   ├── 🐍 dataset.py          # Tải và tiền xử lý dữ liệu MRI
│   ├── 🧠 model.py            # Kiến trúc ResNet50 (BrainTumorClassifier)
│   └── 🏋️ train.py            # Huấn luyện và đánh giá mô hình
│
├── 📂 model/
│   ├── 🤖 knn_model.pkl       # Mô hình KNN đã huấn luyện
│   ├── 🤖 svm_model.pkl       # Mô hình SVM đã huấn luyện
│   └── 📉 pca_model.pkl       # Mô hình PCA giảm chiều
│
├── 📂 data/
│   ├── 📂 train/              # Dữ liệu huấn luyện (MRI images)
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── 📂 test/               # Dữ liệu kiểm tra
│
├── 📂 Brain-Tumor-Test-Images/ # Ảnh mẫu để test nhanh
│
├── 📂 templates/
│   ├── DiseaseDet.html         # Trang chủ
│   ├── uimg.html              # Trang upload ảnh
│   ├── pred.html              # Trang hiển thị kết quả
│   └── error.html             # Trang lỗi
│
├── 📂 static/                 # CSS, JS, ảnh tĩnh
├── 📂 upload/                 # Thư mục lưu ảnh upload
├── 🐍 app.py                  # Flask web application chính
├── 🧪 test.py                 # Script kiểm tra mô hình
└── 📋 requirements.txt        # Danh sách thư viện cần thiết
```

---

## ⚙️ Cài Đặt & Chạy Dự Án

### 📋 Yêu Cầu Hệ Thống

- Python **3.8+**
- CUDA **11.8+** *(khuyến nghị cho GPU)* hoặc CPU
- RAM tối thiểu **8GB**

### 🚀 Hướng Dẫn Cài Đặt

**Bước 1: Clone repository**
```bash
git clone https://github.com/<your-username>/brain_detection.git
cd brain_detection
```

**Bước 2: Tạo môi trường ảo**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**Bước 3: Cài đặt thư viện**

> ⚠️ **Nếu có GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask scikit-learn pillow numpy joblib
```

> 💻 **Nếu chỉ dùng CPU:**
```bash
pip install -r requirements.txt
```

**Bước 4: Khởi động ứng dụng**
```bash
python app.py
```

**Bước 5: Mở trình duyệt**
```
http://localhost:5000
```

---

## 🧪 Huấn Luyện Mô Hình

Nếu muốn huấn luyện lại từ đầu:

```bash
# Chuẩn bị dataset
python src/dataset.py

# Huấn luyện mô hình
python src/train.py
```

> 📌 Dataset khuyến nghị: [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
> Cấu trúc thư mục `data/train/` và `data/test/` theo từng lớp phân loại.

---

## 📊 Pipeline Xử Lý Ảnh

```python
# 1. Tiền xử lý ảnh
transform = Compose([
    Resize((224, 224)),       # Chuẩn hóa kích thước
    ToTensor(),                # Chuyển sang tensor
    Normalize(                 # Chuẩn hóa theo ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2. Trích xuất đặc trưng với ResNet18
# Output: vector 512 chiều

# 3. Giảm chiều với PCA
# Tăng tốc độ & giảm nhiễu

# 4. Phân loại song song
# ├── KNN → class + confidence
# └── SVM → class + confidence
```

---

## 🖥️ Giao Diện Web

-  ![image](https://github.com/user-attachments/assets/d89bc976-4acc-4979-8706-3d0cd35b5751)



| Trang | Mô tả |
|-------|-------|
| `/` | Trang chủ giới thiệu hệ thống |
| `/uimg` (GET) | Form upload ảnh MRI |
| `/uimg` (POST) | Xử lý và trả về kết quả phân loại |

### 📤 Định Dạng Ảnh Được Hỗ Trợ

```
✅ PNG    ✅ JPG / JPEG    ✅ GIF
```
> 📦 Kích thước tối đa: **16 MB**

---

## 🔬 Chi Tiết Kỹ Thuật

### Feature Extractor — ResNet18

```python
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()   # Bỏ lớp classification cuối
resnet.eval()               # Chế độ inference
```

### BrainTumorClassifier — ResNet50 (Fine-tuned)

```python
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=2):
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Fine-tune toàn bộ layers
        self.resnet.fc = nn.Linear(n_inputs, num_classes)
```

### Dual Classification Pipeline

| Model | Ưu điểm | Nhược điểm |
|-------|---------|------------|
| **KNN** | Đơn giản, giải thích được | Chậm với dữ liệu lớn |
| **SVM** | Mạnh với không gian cao chiều | Cần điều chỉnh hyperparameter |

---

## 📈 Kết Quả & Đánh Giá

> Kết quả được hiển thị trực tiếp trên giao diện web gồm:
- 🖼️ Ảnh gốc upload
- 🔲 Ảnh KNN với bounding box và nhãn
- 🔲 Ảnh SVM với bounding box và nhãn
- 📊 Xác suất chi tiết cho 4 lớp: Glioma, Meningioma, No Tumor, Pituitary

---

## 🤝 Đóng Góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. 🍴 Fork repository này
2. 🌿 Tạo branch mới: `git checkout -b feature/ten-tinh-nang`
3. 💾 Commit thay đổi: `git commit -m 'feat: Thêm tính năng X'`
4. 📤 Push lên branch: `git push origin feature/ten-tinh-nang`
5. 📬 Tạo Pull Request

---

## 👨‍💻 Tác Giả

<div align="center">

| | Thông tin |
|---|---|
| 👤 **Họ tên** | Huỳnh Văn An |
| 🎓 **MSSV** | 2200011724 |
| 🏫 **Trường** | Đại học ... |
| 📧 **Email** | *(thêm email của bạn)* |
| 🔗 **GitHub** | *(thêm GitHub của bạn)* |

</div>

---

## 📄 Giấy Phép

Dự án này được phân phối dưới giấy phép **MIT License**. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 📚 Tài Liệu Tham Khảo

- [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn: KNN & SVM](https://scikit-learn.org/stable/)

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy cho một ngôi sao nhé! ⭐**

<img src="https://img.shields.io/github/stars/yourusername/brain_detection?style=social"/>

*Made with ❤️ and 🧠 by Huỳnh Văn An*

</div>
