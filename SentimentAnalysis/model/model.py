import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords + lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# --------------------------------------------------
# 1. LOAD CLEANED DATA
# --------------------------------------------------

df = pd.read_csv("../data/clean_Sentiment.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. CHECK DATA
# --------------------------------------------------

print("\nMissing values:")
print(df[["Text", "Sentiment"]].isnull().sum())

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())


# --------------------------------------------------
# 3. INPUT AND TARGET
# --------------------------------------------------

X = df["Text"]
y = df["Sentiment"]


# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. TF-IDF VECTORIZATION
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# --------------------------------------------------
# 6. TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight = "balanced"
)

model.fit(X_train_tfidf, y_train)


# --------------------------------------------------
# 7. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n-----------------------------")
print("MODEL PERFORMANCE")
print("-----------------------------")

print("Accuracy:", accuracy)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["negative", "neutral", "positive"],
    yticklabels=["negative", "neutral", "positive"]
)

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("Confusion Matrix")

plt.show()


# --------------------------------------------------
# 9. TEST WITH NEW TEXT
# --------------------------------------------------

print("\n" + "-" * 30)
print("TEST YOUR OWN TEXT")
print("-" * 30)

while True:

    text = input("\nEnter a sentence (or type 'exit'): ")

    if text.lower() == "exit":
        print("Exiting...")
        break

    # Clean input exactly like training data
    cleaned_text = clean_text(text)

    # Convert cleaned text to TF-IDF
    text_vector = vectorizer.transform([cleaned_text])

    # Predict sentiment
    prediction = model.predict(text_vector)[0]

    # Get probabilities
    probabilities = model.predict_proba(text_vector)[0]

    # Confidence
    confidence = max(probabilities) * 100

    print("\nPredicted Sentiment:", prediction.upper())
    print("Confidence: {:.2f}%".format(confidence))