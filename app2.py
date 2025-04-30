# 🎨 Fake News Detection Streamlit App with Aesthetic UI

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    fake_df['label'] = 1
    true_df['label'] = 0
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\s+', ' ', text)
    return text

# Header
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #555;
    }
    .footer {
        text-align: center;
        font-size: 15px;
        margin-top: 50px;
        color: gray;
    }
    </style>
    <div class="main-title">📰 Fake News Detection using AI</div>
    <div class="subtitle">Enter a news article and let AI determine if it's real or fake!</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner("Loading model and preparing data..."):
    df = load_data()
    df['text'] = df['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    model = LogisticRegression()
    model.fit(X, y)

# Input
st.markdown("### ✏️ Paste News Text Below:")
user_input = st.text_area("", height=250, placeholder="Paste or type the news article content here...")

# Prediction
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.error("🚨 This news is likely **FAKE**.")
        else:
            st.success("✅ This news is likely **REAL**.")

# About section
with st.expander("ℹ️ About this app"):
    st.write("""
        This is a simple Fake News Detection app powered by **Logistic Regression** and **TF-IDF Vectorizer**.
        
        The model was trained on a dataset of fake and real news articles using classical machine learning techniques.
        
        Developed with ❤️ using Python and Streamlit.
    """)

# Footer
st.markdown("""
    <div class="footer">Made by Vaishwik Vishwakarma </div>
""", unsafe_allow_html=True)