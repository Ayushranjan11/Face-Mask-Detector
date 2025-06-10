
# 😷 Face Mask Detector — Real-Time Detection with Live Alerts  

---

## 🚀 Project Overview  

This is a real-time face mask detection system that uses a deep learning model to identify whether a person is wearing a face mask.  
It processes a **live video stream** from your webcam, draws **bounding boxes** around detected faces, and provides **instant visual and audible alerts** for compliance checks.

---

## ✨ Key Features  

✅ **Real-Time Detection** — Instant feedback from live webcam feed  
✅ **High Accuracy** — Powered by **MobileNetV2** and transfer learning  
✅ **Visual Feedback** — Green box for masked faces, red box for unmasked faces  
✅ **Audible Alerts** — Sound triggered for faces without masks (with cooldown)  

---

## 🛠️ Technology Stack  

| Technology      | Usage                               |
|-----------------|-------------------------------------|
| Python 3.8+     | Core programming language           |
| TensorFlow + Keras | Deep learning model & training     |
| OpenCV          | Real-time video capture + face detection |
| MobileNetV2     | Pre-trained lightweight CNN backbone |
| Scikit-learn    | Data preprocessing & splitting      |
| NumPy + Matplotlib | Data manipulation & visualization  |
| Playsound       | Plays audible alerts on detection    |

---

## 🧠 Methodology  

### 1️⃣ Data Collection & Preprocessing  

- Dataset: [Face Mask Dataset on Kaggle](https://www.kaggle.com/)  
- Images resized to **224x224** pixels  
- Preprocessing with `preprocess_input`  
- Labels one-hot encoded (`with_mask` → `[1,0]`, `without_mask` → `[0,1]`)  

### 2️⃣ Model Architecture & Training  

- **Transfer Learning** with **MobileNetV2**  
- Custom Head:
  - `AveragePooling2D` → `Flatten` → `Dense(128, ReLU)` → `Dropout(0.5)` → `Dense(2, Softmax)`  
- Base model layers frozen  
- Optimizer: **Adam**  
- Loss: **Binary Cross-Entropy**  
- Training on Face Mask Dataset  

### 3️⃣ Real-Time Detection Pipeline  

- **Face Detection:** Haar Cascade Classifier  
- **ROI Extraction:** Detected faces passed to trained model  
- **Classification:** Model predicts `with_mask` or `without_mask`  
- **Feedback:**  
  - Bounding box color (green/red)  
  - Label with confidence score  
  - Audible alert when `without_mask` detected (cooldown enabled)  

---

## ⚙️ Installation & Usage  

### Prerequisites  

- Python 3.8+  
- Git  

### 🔥 Setup Instructions  

#### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/Ayushranjan11/Face-Mask-Detector.git
cd Face-Mask-Detector
```

#### 2️⃣ Create Virtual Environment  

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.env\Scriptsctivate
```

#### 3️⃣ Install Dependencies  

```bash
pip install -r requirements.txt
```

#### 4️⃣ Download Dataset & Pre-trained Model  

- Download the dataset from Kaggle and place it in the `dataset/` folder in the project root.  
- The **Haar Cascade** file `haarcascade_frontalface_default.xml` is already included under `cascades/`.  

#### 5️⃣ Run Real-Time Face Mask Detector  

```bash
python detect_mask_video.py
```

- A webcam window will open.  
- Press **`q`** to quit.  

#### 6️⃣ (Optional) Train the Model From Scratch  

```bash
python train_model.py
```

- Generates:
  - `face_mask_detector.h5` — Trained model  
  - `plot.png` — Training history visualization  

---

## 📈 Results  

The model achieves **high accuracy** on the validation set after **20 epochs**.  
The training plot shows no significant overfitting — training and validation accuracy curves are close.  

<p align="center">
  <img src="plot.png" alt="Training Plot" width="600"/>
</p>


