# Spam Classification with Multiple NLP Approaches

This project demonstrates and compares three different approaches for binary text classification (**spam** vs. **ham**). It includes end-to-end workflows from preprocessing to model evaluation.

---

## 🔍 Overview

The following classification pipelines are implemented:

1. **TF-IDF + Naive Bayes**  
   A classic baseline using `TfidfVectorizer` and `MultinomialNB` from scikit-learn.

2. **FastText + KNN**  
   Texts are embedded using FastText, followed by K-Nearest Neighbors classification.

3. **BERT Embeddings + Feedforward Neural Network (FNN)**  
   Sentence embeddings are generated using `sentence-transformers` (`bert-base-uncased`), then passed to an MLP classifier.
