# Sentiment Analysis

A complete **Sentiment Analysis** project built with Python that uses both **traditional Machine Learning** and **Deep Learning** techniques to classify text into three sentiment categories: **Positive, Negative, and Neutral**.

The project was developed to compare two different approaches to text classification: **TF-IDF with Logistic Regression** and a **PyTorch-based LSTM neural network**. It includes data preprocessing, feature extraction, model training, evaluation, visualization, model saving, and an interactive system for testing custom sentences.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine the emotional tone or opinion expressed in a piece of text.

This project takes a text sentence as input and predicts whether its sentiment is:

* 🟢 **Positive**
* ⚪ **Neutral**
* 🔴 **Negative**

For example:

```text
"I absolutely love this product!"
→ POSITIVE

"The product is okay."
→ NEUTRAL

"I really hate this product."
→ NEGATIVE
```

The project contains two different models so their approaches can be compared:

### 1. Logistic Regression

A traditional Machine Learning approach using **TF-IDF** to convert text into numerical features.

### 2. PyTorch LSTM

A Deep Learning approach using an **Embedding layer and Bidirectional LSTM** to learn patterns from sequences of words.

---

# 🚀 Features

* Text data preprocessing
* Missing-value handling
* Sentiment distribution analysis
* Train/test splitting
* TF-IDF feature extraction
* Logistic Regression classification
* PyTorch LSTM classification
* Bidirectional LSTM architecture
* Class-weighted training
* Validation during training
* Model evaluation
* Accuracy calculation
* Precision, Recall and F1-score
* Confusion matrix visualization
* Training/validation accuracy graphs
* Training/validation loss graphs
* Interactive custom-text prediction
* Prediction confidence/probabilities
* Trained model saving using `.pth`
* Requirements file for easy installation

---

# 🧠 Machine Learning Approach

The first model uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to represent text numerically.

The basic workflow is:

```text
Raw Text
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Sentiment Prediction
```

TF-IDF gives higher importance to words that are useful for distinguishing between different documents while reducing the importance of very common words.

The project also uses **unigrams and bigrams**, allowing the model to consider both individual words and word pairs.

For example:

```text
"very good"
"not good"
"really bad"
```

can provide useful information to the classifier.

---

# 🔥 Deep Learning Approach

The second model uses a **PyTorch Bidirectional LSTM**.

The workflow is:

```text
Raw Text
   ↓
Preprocessing
   ↓
Tokenization
   ↓
Numerical Sequences
   ↓
Padding
   ↓
Embedding
   ↓
Bidirectional LSTM
   ↓
Dropout
   ↓
Fully Connected Layer
   ↓
Softmax
   ↓
Sentiment
```

### Model Architecture

The LSTM model consists of:

```text
Embedding
    ↓
Bidirectional LSTM
    ↓
Dropout
    ↓
Dense / Fully Connected Layer
    ↓
Dropout
    ↓
Output Layer
```

The output layer contains three classes:

```text
Negative
Neutral
Positive
```

The model produces probabilities for each class and selects the class with the highest probability.

---

# 📊 Model Evaluation

The models are evaluated using several metrics rather than relying only on accuracy.

### Accuracy

Measures the percentage of predictions that were classified correctly.

### Precision

Measures how many predictions for a particular class were actually correct.

### Recall

Measures how many samples belonging to a class were correctly identified.

### F1-Score

Combines precision and recall into a single metric.

### Confusion Matrix

The confusion matrix provides a detailed view of correct and incorrect predictions for each sentiment class.

Example:

```text
                 Predicted
              Negative Neutral Positive

Actual Negative    ✓       ✗       ✗
       Neutral     ✗       ✓       ✗
       Positive    ✗       ✗       ✓
```

---

# 📈 Visualization

The project generates visualizations to understand model performance.

### Confusion Matrix

Shows how predictions are distributed between:

* Negative
* Neutral
* Positive

### Training vs Validation Accuracy

Used to observe whether the model is learning effectively and to identify potential overfitting.

### Training vs Validation Loss

Shows how the training and validation losses change during training.

These visualizations make it easier to understand the behavior of the deep learning model during training.

---

# 🗂️ Project Structure

```text
SentimentAnalysis/
│
├── clean/
│   └── clean.py
│
├── data/
│   ├── Sentiment_dataset.csv
│   └── clean_Sentiment.csv
│
├── model/
│   ├── model.py
│   ├── model_lstm.py
│   └── best_lstm_model.pth
│
├── requirements.txt
│
└── README.md
```

### `clean/`

Contains scripts used for cleaning and preparing the dataset.

### `data/`

Contains the original and processed datasets.

### `model/`

Contains the Machine Learning and LSTM implementations along with the saved PyTorch model.

### `requirements.txt`

Contains the Python libraries required to run the project.

---

# 🛠️ Technologies Used

### Programming Language

* **Python**

### Data Processing

* **Pandas**
* **NumPy**

### Machine Learning

* **Scikit-learn**
* TF-IDF
* Logistic Regression
* Train/Test Split
* Classification Metrics

### Natural Language Processing

* **NLTK**
* Stopword handling
* Word lemmatization
* Text preprocessing

### Deep Learning

* **PyTorch**
* Embedding
* LSTM
* Bidirectional LSTM
* Dropout
* Cross-Entropy Loss

### Visualization

* **Matplotlib**
* **Seaborn**

---

# ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd SentimentAnalysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 📚 NLTK Resources

The project uses NLTK resources for text preprocessing.

The required resources can be downloaded using:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

# ▶️ Running the Project

After installing the dependencies, run the desired model from the project directory.

For example:

```bash
python model/model.py
```

For the PyTorch LSTM:

```bash
python model/model_lstm.py
```

Make sure the dataset paths in the Python files match the structure of your project.

---

# 💬 Testing Custom Sentences

The LSTM model includes an interactive prediction system.

After the model finishes training, you can enter your own sentence:

```text
Enter a sentence (or type 'exit'): I love this product
```

The model returns something similar to:

```text
Predicted Sentiment: POSITIVE

Probabilities:

Negative: 2.31%
Neutral: 4.85%
Positive: 92.84%
```

You can continue entering sentences until:

```text
exit
```

is entered.

---

# 💾 Saved Model

The trained PyTorch model can be saved as:

```text
best_lstm_model.pth
```

This allows the trained model to be reused without having to train it from scratch every time.

The model can later be loaded for prediction in another Python application.

---

# 🔍 Important Note

The performance of a sentiment-analysis model depends heavily on the **quality, size, and diversity of the training dataset**.

A model trained on a relatively small dataset may perform well on examples similar to its training data but may struggle with completely new wording, slang, sarcasm, or complex contextual sentences.

Therefore, the evaluation metrics and confusion matrix should be considered together when judging the model's performance.

---

# 🎯 Learning Objectives

This project helped explore several important concepts in Machine Learning and Deep Learning, including:

* Data cleaning and preprocessing
* Exploratory analysis of datasets
* Text classification
* Feature engineering
* TF-IDF representation
* Logistic Regression
* Tokenization
* Word embeddings
* Sequence padding
* Recurrent Neural Networks
* LSTM networks
* Bidirectional LSTM
* Model evaluation
* Confusion matrices
* Overfitting and validation
* Saving and loading trained models
* Building an interactive prediction system

---

# 🔮 Future Improvements

Possible improvements for the project include:

* Increasing the size and diversity of the dataset
* Improving text preprocessing
* Experimenting with different LSTM architectures
* Hyperparameter tuning
* Using pretrained word embeddings
* Comparing additional Machine Learning algorithms
* Adding a web interface for predictions
* Creating an API for the trained model
* Deploying the sentiment-analysis model online
* Experimenting with Transformer-based models such as BERT

---

# 👨‍💻 Author

**Mohd Taufique Alam**

This project was developed as a practical Machine Learning and Deep Learning project to explore **sentiment classification using both traditional Machine Learning and PyTorch-based LSTM models**.
