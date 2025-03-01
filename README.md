Amazon Reviews Sentiment Analysis

📌 Overview

This project performs sentiment analysis on Amazon reviews using two different approaches:

VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK, a rule-based model that scores sentiment using a bag-of-words approach.

RoBERTa, a transformer-based model from Hugging Face, which understands context and nuances in text more effectively.
The goal is to compare the accuracy and effectiveness of these models in classifying review sentiment.

✨ Features - 

✔ Sentiment Analysis with NLTK's VADER – Quick and efficient rule-based sentiment scoring.

✔ Deep Learning Approach with RoBERTa – Context-aware classification using a state-of-the-art transformer model.

✔ Comparison of Model Outputs – See how different models interpret sentiment.

✔ Data Visualization – Explore sentiment trends and correlations with review ratings.

✔ Error Analysis – Examine cases where sentiment models fail or produce unexpected results.

✔ Hugging Face Pipelines Integration – Simplified sentiment classification with minimal setup.

🔧 Installation

Ensure you have Python installed, then install the required dependencies:

pip install nltk transformers datasets torch matplotlib seaborn

Download the necessary NLTK data:

import nltk
nltk.download('vader_lexicon')

🚀 Usage

1️⃣ Sentiment Analysis with VADER

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

text = "This product is amazing! I love it."

score = sia.polarity_scores(text)

print(score)

2️⃣ Sentiment Analysis with RoBERTa

from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

result = sentiment_pipeline("This product is amazing! I love it.")

print(result)

📊 Results & Insights

VADER is fast and works well for short, simple text but struggles with sarcasm and nuanced sentiment.

RoBERTa captures sentiment more accurately, understanding context better but requiring more computational power.
Visualizing results helps understand how sentiment aligns with review ratings.

🔮 Next Steps

✅ Expand the dataset for better model evaluation.

✅ Fine-tune RoBERTa on Amazon-specific reviews.

✅ Experiment with other transformer models like BERT or DistilBERT.


Author : Yatendra Upadhyay
