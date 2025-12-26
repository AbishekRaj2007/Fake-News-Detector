# 📰 Fake News Detector

A Machine Learning–based Fake News Detection application that classifies news text as **Fake** or **Real** using Natural Language Processing (NLP).  
The trained model and vectorizer are saved and used directly for predictions via a Python application.

---

## 📌 Project Description

Fake news spreads rapidly across digital platforms and can cause serious social impact.  
This project aims to detect fake news articles by analyzing textual content using a trained machine learning model.

The system takes news text as input, transforms it using a trained vectorizer, and predicts whether the news is **Fake** or **Real**.

---

## 🧠 Features

- Text-based fake news detection
- Pre-trained machine learning model
- Saved vectorizer for consistent text transformation
- Simple and lightweight Python application
- Fast prediction without retraining the model

---

## 🛠️ Technologies Used

- **Language:** Python  
- **Libraries:**
  - NumPy
  - Pandas
  - Scikit-learn
  - Pickle
  - NLP libraries (NLTK / similar)

---

## 📂 Project Structure

Fake-News-Detector/
│
├── app.py # Main application file for prediction
├── model.pkl # Trained machine learning model
├── vectorizer.pkl # Saved text vectorizer
├── tnesor.py # Model-related logic / experimentation file
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ How It Works

1. User provides news text as input  
2. Text is transformed using the saved vectorizer  
3. The trained model processes the transformed data  
4. Output is predicted as **Fake** or **Real**

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
2️⃣ Install Dependencies
bash
Copy code
pip install numpy pandas scikit-learn
3️⃣ Run the Application
bash
Copy code
python app.py
