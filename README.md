# 📧 SMS Spam Detection System

A Machine Learning and Natural Language Processing (NLP) based application that classifies SMS messages as **Spam** or **Ham (Not Spam)**.

The project uses text preprocessing, TF-IDF vectorization, and a Multinomial Naive Bayes classifier to accurately identify unwanted messages.

---

## 🚀 Features

- SMS spam classification
- Text preprocessing using NLTK
- TF-IDF feature extraction
- Multinomial Naive Bayes model
- High precision spam detection
- Saved model and vectorizer for deployment

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- Matplotlib
- Pickle

---

## 📂 Project Workflow

```text
Raw SMS Message
       │
       ▼
Text Preprocessing
       │
       ▼
Tokenization
       │
       ▼
Stopword Removal
       │
       ▼
Stemming
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Multinomial Naive Bayes
       │
       ▼
Spam / Ham Prediction
```

---

## 🔍 Text Preprocessing

Each message undergoes several preprocessing steps before being passed to the model:

### 1. Lowercasing

Converts all characters to lowercase.

**Example**

```text
WIN A FREE iPhone NOW!
↓
win a free iphone now!
```

### 2. Tokenization

Splits the text into individual words.

```text
win a free iphone now
↓
['win', 'a', 'free', 'iphone', 'now']
```

### 3. Removing Special Characters

Keeps only alphanumeric tokens.

```text
['win', 'free', 'iphone', '!!!']
↓
['win', 'free', 'iphone']
```

### 4. Stopword Removal

Removes common English words such as:

```text
a, an, the, is, are, and, of, in...
```

### 5. Stemming

Reduces words to their root forms using Porter Stemmer.

```text
playing → play
played  → play
plays   → play
```

---

## 📊 Exploratory Data Analysis

Before training the model, several message characteristics were analyzed:

- Number of characters
- Number of words
- Number of sentences

This helped identify patterns between spam and legitimate messages and provided insights into dataset distribution.

---

## 🧠 Feature Engineering

The cleaned text is converted into numerical representations using:

### TF-IDF Vectorization

```python
TfidfVectorizer(max_features=3000)
```

TF-IDF helps:

- Capture important words
- Reduce the impact of common words
- Improve classification performance
- Generate meaningful sparse vectors

---

## 🤖 Model Training

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

Different Naive Bayes variants were explored:

- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Multinomial Naive Bayes

### Final Model

**Multinomial Naive Bayes**

Chosen because it performed best for text classification with TF-IDF features.

---

## 📈 Evaluation Metrics

Model performance was evaluated using:

- Accuracy Score
- Precision Score
- Confusion Matrix

Special focus was placed on **Precision**, since minimizing false spam predictions is important in real-world applications.

---

## 💾 Saved Artifacts

The trained model and vectorizer are stored using Pickle:

```text
model.pkl
vectorizer.pkl
```

These files can be loaded directly during deployment without retraining the model.

---

## 🎯 Key Learnings

- Natural Language Processing fundamentals
- Text preprocessing techniques
- Feature extraction with TF-IDF
- Text classification using Naive Bayes
- Machine Learning model evaluation
- Model serialization and deployment readiness

---

## 🔮 Future Improvements

- Deep Learning models (LSTM, GRU)
- Transformer-based architectures (BERT)
- Email spam detection
- Multilingual spam classification
- Explainable AI for prediction insights

---
