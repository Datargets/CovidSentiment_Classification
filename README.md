# Sentiment Classification of COVID-19 Tweets Using Transformer Embeddings and Classical ML Models
📌 Description
This project presents a hybrid approach combining pretrained transformer-based language models—BERT, DistilBERT, and XLM-RoBERTa—with traditional machine learning classifiers such as Support Vector Machine (SVM) and Logistic Regression for sentiment classification of COVID-19 tweets. The project also evaluates the performance of zero-shot learning using Hugging Face pipelines for comparison.

📂 Dataset Information
Source: Kaggle COVID-19 Twitter Dataset

Time Period: April to June 2020

Labels: positive, neutral, negative

Sample Size: 7,500 tweets (2,500 per class)

Preprocessing:

Lowercasing

Removing URLs, hashtags, mentions, digits, and punctuation

Tokenization and stopword removal

🧠 Methodology
1. Data Preparation
Stratified sampling for balanced class representation.

Label encoding using LabelEncoder.

Splitting into training (80%) and testing (20%) sets.

2. Embedding Generation
Models used: bert-base-uncased, distilbert-base-uncased, xlm-roberta-base

Embeddings extracted using:

Mean pooling of last hidden states or [CLS] token

Time for embedding generation:

BERT: 125.86 sec

DistilBERT: 76.45 sec

RoBERTa: 141.05 sec

3. Feature Scaling
Standardization with StandardScaler to enhance SVM performance.

4. Classification Models
SVM (RBF kernel, C=0.1)

Logistic Regression (C=0.001)

Evaluation using classification_report (precision, recall, F1-score, accuracy)

5. Zero-Shot Classification
Hugging Face pipeline() used with same transformer models (without fine-tuning).

Labels: positive, neutral, negative

💻 Code Information
Language: Python 3.10+

Libraries:

transformers

sklearn

torch

pandas, numpy

tqdm

Scripts:

preprocess.py: Cleans and prepares the tweet data

embedding_extraction.py: Uses transformer models to generate sentence embeddings

train_classifiers.py: Trains SVM and Logistic Regression models

evaluate_models.py: Evaluates models using sklearn metrics

zero_shot.py: Performs zero-shot sentiment classification


⚙️ Requirements
transformers==4.40.0
scikit-learn
torch>=2.0
pandas
numpy
tqdm

