# 🔢 MNIST Digit Classifier

> Handwritten digit recognition powered by a Keras Dense Neural Network — deployed via FastAPI on HuggingFace Spaces + Streamlit Cloud frontend.

[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi)](https://huggingface.co/spaces/YOUR_HF_USERNAME/mnist-api)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat&logo=streamlit)](https://YOUR_APP.streamlit.app)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python)](https://python.org)

---

## 🚀 Live Demo

| Service | URL |
|---------|-----|
| 🌐 Streamlit Frontend | `https://mnist-digit-predictor-by-muhammad-taha.streamlit.app/` |
| ⚡ FastAPI Backend | `https://huggingface.co/spaces/funterpie/MNIST_Digit_predictor` |
| 📖 API Swagger Docs | `https://huggingface.co/spaces/funterpie/MNIST_Digit_predictor/docs` |

---






## 🧠 Model Architecture

```
Input Layer     →  784 neurons (28×28 flattened)
Dense Layer 1   →  256 neurons, ReLU activation
Dense Layer 2   →  128 neurons, ReLU activation
Output Layer    →  10 neurons, Softmax activation
```

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (70,000 images) |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 10 |
| Batch Size | 128 |
| **Test Accuracy** | **97.65%** |

---

## ⚡ API Reference

### Health Check
```http
GET /
```
```json
{
  "message": "MNIST Digit Classifier API is running.",
  "docs": "/docs"
}
```

### Predict Digit
```http
POST /predict/
Content-Type: multipart/form-data
```

**Request:** Upload any PNG/JPG image of a handwritten digit.

**Response:**
```json
{
  "filename": "digit_3.png",
  "predicted_digit": 3,
  "confidence": 0.9921,
  "all_probabilities": {
    "0": 0.0001, "1": 0.0000, "2": 0.0012,
    "3": 0.9921, "4": 0.0003, "5": 0.0020,
    "6": 0.0001, "7": 0.0002, "8": 0.0035, "9": 0.0005
  }
}
```

### cURL Example
```bash
curl -X POST "https://YOUR_HF_USERNAME-mnist-api.hf.space/predict/" \
  -F "file=@digit_0.png"
```

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/funterpie/mnist-digit-classifier.git
cd mnist-digit-classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run FastAPI backend
uvicorn main:app --reload
# → http://127.0.0.1:8000/docs

# 5. Run Streamlit frontend (new terminal)
pip install streamlit
streamlit run app.py
# → http://localhost:8501
```

---

## ☁️ Deployment Guide

### FastAPI → HuggingFace Spaces (Docker)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as SDK
3. Push these files:
   - `main.py`
   - `mnist_model.keras`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`

```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/mnist-api
cd mnist-api
# copy your files here
git add . && git commit -m "deploy: FastAPI MNIST backend"
git push
```

### Streamlit Frontend → Streamlit Cloud

1. Push `app.py` + `requirements.txt` + `sample_digits/` to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Update `API_URL` in `app.py` to your HF Space URL

---

## 👨‍💻 Author

**Muhammad Taha Sattar Arain**
Founder & CEO — [Alpha Orbit](https://alphaorbit.site)
SMIT Batch 10 — AI & Data Science

[![Email](https://img.shields.io/badge/Email-taha@alphaorbit.site-blue?style=flat&logo=gmail)](mailto:taha@alphaorbit.site)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-taha--arain-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/taha-arain)
[![GitHub](https://img.shields.io/badge/GitHub-funterpie-181717?style=flat&logo=github)](https://github.com/funterpie)

---

*Assignment 8 — Deep Learning | SMIT AI & Data Science Batch 10*
