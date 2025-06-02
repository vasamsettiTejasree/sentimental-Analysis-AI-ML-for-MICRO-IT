# sentimental-Analysis-AI-ML-for-MICRO-IT



# 💬 Twitter Sentiment Analysis using Machine Learning

> A robust and scalable machine learning pipeline for real-time Twitter sentiment analysis, leveraging Natural Language Processing (NLP) and Naive Bayes classification.

---

## 📌 Overview

This project aims to classify tweets as **Positive** or **Negative** using a clean and interpretable machine learning pipeline. Built using **Python**, it incorporates powerful NLP techniques like **TF-IDF** vectorization and **lemmatization**, and applies the **Multinomial Naive Bayes** algorithm for high-accuracy sentiment classification.

---

## 🧠 Key Features

- 🔍 **Data Cleaning**: Noise-free tweet preprocessing (URL removal, mentions, special characters).
- 🧽 **Text Normalization**: Lowercasing, stopwords removal, and lemmatization.
- 🧠 **TF-IDF Feature Extraction**: Captures the most relevant words and phrases using n-grams.
- 🤖 **Model Training**: Multinomial Naive Bayes classifier optimized for textual data.
- 📊 **Model Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix visualization.
- 🧾 **Real-Time Predictions**: Instantly predict the sentiment of custom tweets.
- 📈 **Visualization**: Clear sentiment distribution and heatmaps for evaluation.

---

## 🗃️ Dataset

- **Source**: [Twitter Sentiment Dataset](https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv)
- **Classes**: 
  - `0` → Negative  
  - `1` → Positive
- **Columns**: 
  - `label`: Sentiment class  
  - `tweet`: Raw tweet text

---

## 🚀 Getting Started

### 🔧 Prerequisites

Ensure you have Python 3.x installed.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk

## Download NLTK Resources

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

## Run the Script
 sentiment_analysis.py

## Sample Prediction

Input: "I absolutely love the features of this app!"
Output: Predicted Sentiment → Positive

##Future Enhancements
Add Neutral sentiment class for multi-class classification.

Experiment with advanced models like SVM, Logistic Regression, or LSTM/Transformers.

Deploy as a web application using Streamlit or Flask.

Integrate Twitter API for live sentiment monitoring.


