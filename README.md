# 📊 Tweet Sentiment Analysis Using Deep Learning and FastAPI

## 📝 Overview
The **Tweet Sentiment Analysis** project is a machine learning system that classifies tweets as **disaster-related or not** using **Deep Learning (TensorFlow, Universal Sentence Encoder)**. It is deployed via **FastAPI** for real-time inference.

## 🚀 Features
- **Tweet Classification 📢**: Predicts whether a tweet is related to a disaster or not.
- **Deep Learning Model 🤖**: Trained using **TensorFlow** and **Universal Sentence Encoder**.
- **FastAPI-Based API ⚡**: Deploys a RESTful API for real-time sentiment prediction.
- **Flask-Based Alternative API 🔥**: Provides an optional Flask implementation.
- **JSON-Based API Response 📄**: Accepts JSON input and returns classification results with confidence scores.

## 🛠 Tech Stack
- **Natural Language Processing (NLP) 🧠**: Universal Sentence Encoder (USE) for text representation.
- **Machine Learning Frameworks 📊**: TensorFlow & Keras.
- **API Development 🚀**: FastAPI & Flask.
- **Deployment 🖥️**: Uvicorn for FastAPI, Postman for API testing.

## ⚙️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rpraveen760/DLSentimentAnalysis.git
cd DLSentimentAnalysis
```

### 2️⃣ Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3️⃣ Run FastAPI Server
```bash
uvicorn main:app --reload
```

API will be live at: `http://127.0.0.1:8000/`

### 4️⃣ Run Flask Server (Alternative)
```bash
python deploy_tweet_classifier.py
```

## 📡 API Usage (FastAPI)
### **POST /predict/** (Predict if a tweet is disaster-related)
#### **Request**:
```json
{
    "text": "Massive flooding in New York, thousands evacuated."
}
```
#### **Response**:
```json
{
    "tweet": "Massive flooding in New York, thousands evacuated.",
    "is_disaster": true,
    "confidence": 0.8765
}
```

## 📂 Project Structure
```
📦 DLSentimentAnalysis
 ┣ 📂 deploy_tweet_classifier.py    # Flask API
 ┣ 📂 main.py                      # FastAPI API
 ┣ 📂 helper_functions.py           # Utility functions
 ┣ 📂 text_vectorizer/              # TensorFlow text preprocessing
 ┣ 📂 requirements.txt              # Dependencies
 ┣ 📂 README.md                     # Documentation
 ┣ 📂 .gitignore                     # Excluded files
 ┗ 📂 model_6.h5 (NOT uploaded, hosted externally)
```

## 📦 Model 
Since the trained model is too large for GitHub, it wasn't uploaded

