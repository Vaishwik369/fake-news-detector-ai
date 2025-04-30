# fake-news-detector-ai
An interactive Streamlit web app for detecting fake news using Logistic Regression and TF-IDF. Built with Python and trained on real-world news datasets.
# 📰 Fake News Detector — Powered by AI | Streamlit App

An interactive and aesthetic web app built using **Streamlit** to detect whether a given news article is **Fake** or **Real** using a machine learning model trained on real-world data.

---

## 🚀 Features

- 🔍 Paste any news text to check its authenticity
- 🧠 AI-powered Logistic Regression model with TF-IDF vectorization
- 🎨 Aesthetic UI using Streamlit
- 📊 Real-time prediction and feedback
- 📁 Based on a curated dataset of fake and real news articles

---

## 🧪 Tech Stack

- **Python 3**
- **Streamlit**
- **Scikit-learn**
- **Pandas / NumPy**
- **TF-IDF Vectorizer**
- **Logistic Regression**

---

## 📁 Dataset

- **Fake.csv**: News articles labeled as Fake
- **True.csv**: News articles labeled as Real
- Dataset Source: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detector-ai.git
cd fake-news-detector-ai
▶️ Run Locally
bash
Copy
Edit
streamlit run app.py
Make sure Fake.csv and True.csv are in the same directory.

🧠 Model Training
Model training (Logistic Regression, XGBoost, Random Forest, LSTM) is included in the fake_news_training.py script.
However, the web app uses the lightweight Logistic Regression model for fast real-time inference.
