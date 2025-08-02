## Sentiment Classification of COVID-19 Tweets Using Transformer Embeddings and Classical Machine Learning Models
📌 Description

This project presents a hybrid sentiment analysis framework that integrates pretrained transformer-based language models (BERT, DistilBERT, XLM-RoBERTa) with traditional machine learning classifiers (Support Vector Machine and Logistic Regression) to classify sentiment in COVID-19-related tweets. Additionally, it evaluates zero-shot learning performance using Hugging Face pipelines for task-agnostic inference.
📊 Dataset Information

Source: Kaggle COVID-19 Twitter Dataset

Time Period: April to June 2020

Labels: Positive, Neutral, Negative

Sample Size: 7,500 tweets (2,500 per sentiment class using stratified sampling)

Preprocessing Steps:

  Lowercasing all text

  Removing URLs, hashtags, mentions, digits, and punctuation

  Tokenization

  Stopword removal

🧠 Model Architecture Overview

   ![Model Architecture](images/Figure1.png)


🧾 Code Information

   Language: Python 3.10+

   Key Notebooks:

        Bert_Covid_Sentiment_Hybrid.ipynb: BERT + SVM/Logistic Regression

        Covid_Sentiment_Bert.ipynb: BERT embeddings

        Covid_Sentiment_distilBert.ipynb: DistilBERT embeddings

        DistilBert_Covid_Sentiment_Hybrid.ipynb: DistilBERT + SVM/Logistic Regression

        Covid_Sentiment_Roberta.ipynb: XLM-RoBERTa embeddings

        RoBERTACovid_Sentiment_Hybrid.ipynb: XLM-RoBERTa + SVM/Logistic Regression

        README.md: Project description and usage instructions

    Repository: https://github.com/Datargets/CovidSentiment_Classification

💻 Usage Instructions

    Clone the repository:

git clone https://github.com/Datargets/CovidSentiment_Classification.git
cd CovidSentiment_Classification

Install dependencies:

pip install -r requirements.txt

Download and place the dataset:

    Get the dataset from Kaggle.

    Place it in the data/ directory.

Run analysis notebooks:

    Use Jupyter to open .ipynb files and execute them.

View Results:

    Classification metrics (Accuracy, Precision, Recall, F1-Score) and plots are shown inline or saved in results/.

📦 Requirements

    Python Libraries:

        transformers==4.40.0

        scikit-learn

        torch>=2.0

        pandas

        numpy

        tqdm

    Hardware:

        GPU recommended (for faster embeddings)

        Minimum 16 GB RAM

🧪 Methodology
1. Data Preparation

    Balanced dataset: 7,500 tweets (2,500/class)

    Encoded sentiments with LabelEncoder

    Split: 80% training (6,000) / 20% testing (1,500)

2. Embedding Generation

    Models: bert-base-uncased, distilbert-base-uncased, xlm-roberta-base

    Embeddings extracted using:

        Mean pooling of last hidden states

        OR [CLS] token

    Average Generation Time:

        BERT: 125.86s

        DistilBERT: 76.45s

        RoBERTa: 141.05s

3. Feature Scaling

    Standardized with StandardScaler for SVM optimization

4. Classification Models

    SVM (RBF kernel, C=0.1)

    Logistic Regression (C=0.001)

    Evaluation: Accuracy, Precision, Recall, F1-score via classification_report

5. Zero-Shot Classification

    Hugging Face pipeline() without fine-tuning

    Sentiment labels: Positive, Neutral, Negative

📚 Citations

    
    •	Rezaei, Z., Safi Samghabadi, S., & Banad, Y. M. (2025). A Scalable Hybrid Framework for Sentiment Analysis of COVID-19 Tweets Using Transformer Embeddings and Lightweight Classifiers. (Unpublished manuscript).
    •	Dataset: Chakraborty, A. K. (2020). COVID-19 Twitter Dataset. Kaggle. Available at: https://www.kaggle.com/datasets/arunavakrchakraborty/covid19-twitter-dataset.


📜 License & Contributions

    License: MIT License

    Contributions:
    Pull requests and issue reporting are welcome at the GitHub repository.

✅ Conclusions

    The hybrid method combining frozen transformer embeddings with classical ML models balances semantic power and efficiency.

    DistilBERT + Logistic Regression performs best for speed/accuracy trade-off:

        Accuracy: 0.64, F1-score: 0.62

    Zero-shot performance was notably lower (Accuracy: 0.21–0.33), affirming the importance of task-specific adaptation.

⚠️ Limitations

    Small dataset (7,500 tweets) may limit generalization.

    Only English tweets considered—no multilingual evaluation.

    Static time range (April–June 2020) may miss later developments.

    Potential Twitter bias (e.g., bots, demographic skew).

    No fine-tuning of transformers—domain adaptation might improve results.
