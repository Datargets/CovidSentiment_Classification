##Sentiment Classification of COVID-19 Tweets Using Transformer Embeddings and Classical Machine Learning Models
Description
This project introduces a hybrid framework for sentiment classification of COVID-19-related tweets, integrating pretrained transformer-based language models (BERT, DistilBERT, XLM-RoBERTa) with computationally efficient traditional machine learning classifiers (Support Vector Machine and Logistic Regression). The approach leverages transformer embeddings for rich semantic feature extraction and classical classifiers for rapid, interpretable sentiment analysis. Additionally, zero-shot classification using Hugging Face pipelines is evaluated for comparison. The framework is designed to be scalable and efficient, suitable for real-time sentiment monitoring in resource-constrained environments during public health crises.
Dataset Information
Source: Kaggle COVID-19 Twitter Dataset (link)
Time Period: April to June 2020
Labels: Positive, Neutral, Negative
Sample Size: 7,500 tweets (2,500 per sentiment class, balanced via stratified sampling)
Preprocessing Steps:
Lowercasing all text to ensure uniformity
Removal of URLs, hashtags, mentions, digits, and punctuation to reduce noise
Tokenization to segment text into individual tokens
Stopword removal to eliminate common words with low semantic value (e.g., "the," "is," "and")
Code Information
Programming Language: Python 3.10+
Scripts:
preprocess.py: Handles data cleaning and preprocessing of tweet data
embedding_extraction.py: Generates sentence embeddings using transformer models
train_classifiers.py: Trains SVM and Logistic Regression models on extracted embeddings
evaluate_models.py: Evaluates model performance using precision, recall, F1-score, and accuracy
zero_shot.py: Implements zero-shot sentiment classification using Hugging Face pipelines
Repository: Available at https://github.com/Datargets/CovidSentiment_Classification
Usage Instructions
Clone the Repository:
git clone https://github.com/Datargets/CovidSentiment_Classification.git
cd CovidSentiment_Classification
Install Dependencies:
pip install -r requirements.txt
Download the Dataset:
Obtain the dataset from Kaggle.
Place the dataset in the project directory under data/.
Run Preprocessing:
python preprocess.py
Generate Embeddings:
python embedding_extraction.py
Train and Evaluate Models:
python train_classifiers.py
python evaluate_models.py
Perform Zero-Shot Classification:
python zero_shot.py
View Results:
Model performance metrics (precision, recall, F1-score, accuracy) are saved in the results/ directory.
Requirements
Python Libraries:
transformers==4.40.0
scikit-learn
torch>=2.0
pandas
numpy
tqdm
Hardware:
A GPU is recommended for faster embedding generation, though CPU execution is supported.
Minimum 16GB RAM for handling transformer models efficiently.
Methodology
Data Preparation:
Stratified sampling to create a balanced dataset (7,500 tweets, 2,500 per class).
Label encoding of sentiment labels (positive, neutral, negative) using LabelEncoder.
Dataset split: 80% training (6,000 samples) and 20% testing (1,500 samples).
Embedding Generation:
Transformer models (bert-base-uncased, distilbert-base-uncased, xlm-roberta-base) used in a frozen state.
Embeddings extracted via mean pooling of last hidden states or [CLS] token representation.
Embedding generation times:
BERT: 125.86 seconds
DistilBERT: 76.45 seconds
RoBERTa: 141.05 seconds
Feature Scaling:
Standardization of embeddings using StandardScaler to ensure zero mean and unit variance, enhancing SVM performance.
Classification:
Models: SVM (RBF kernel, C=0.1) and Logistic Regression (C=0.001).
Evaluation metrics: Precision, recall, F1-score, and accuracy computed via classification_report.
Zero-Shot Classification:
Implemented using Hugging Face pipeline() with the same transformer models, without fine-tuning.
Labels: Positive, neutral, negative.
Citations
Rezaei, Z., Samghabadi, S. S., & Banad, Y. M. (2025). A Scalable Hybrid Framework for Sentiment Analysis of COVID-19 Tweets Using Transformer Embeddings and Lightweight Classifiers. (Unpublished manuscript).
Dataset: Chakraborty, A. K. (2020). COVID-19 Twitter Dataset. Kaggle. Available at: https://www.kaggle.com/datasets/arunavakrchakraborty/covid19-twitter-dataset.
License & Contribution Guidelines
License: MIT License. See LICENSE file in the repository for details.
Contributions: Contributions are welcome! Please submit pull requests or open issues on the GitHub repository for bug fixes, feature additions, or documentation improvements.
Conclusions
The hybrid framework effectively combines transformer-based embeddings with classical machine learning classifiers, achieving a balance between semantic richness and computational efficiency. The DistilBERT + Logistic Regression model demonstrated the best trade-off, with an accuracy of 0.64 and an F1-score of 0.62 on a balanced dataset of 7,500 tweets. Zero-shot classification, while flexible, yielded poor performance (accuracy 0.21–0.33), underscoring its limitations in supervised settings. The approach is highly modular, allowing easy substitution of embedding models or classifiers, and is well-suited for real-time sentiment monitoring during public health crises.
Limitations
Limited Sample Size: The study uses a subset of 7,500 tweets due to computational constraints, potentially limiting the generalizability of findings compared to the full dataset (143,902 tweets).
Lack of Multilingual Analysis: The framework focuses on English tweets, excluding non-English sentiments that could provide broader insights.
Static Time Frame: Data is restricted to April–June 2020, missing temporal dynamics of sentiment across different pandemic stages.
Potential Biases in Twitter Data: The dataset may not fully represent the general population due to Twitter’s user demographics and potential bot activity.
No Fine-Tuning in Hybrid Approach: While computationally efficient, the frozen transformer models may not capture domain-specific nuances as effectively as fine-tuned models.
