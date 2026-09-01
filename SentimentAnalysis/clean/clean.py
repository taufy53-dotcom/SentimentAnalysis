import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# 1. DOWNLOAD NLTK DATA
# --------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("../data/Sentiment_dataset.csv")

print("Original shape:", df.shape)


# --------------------------------------------------
# 3. REMOVE UNNECESSARY COLUMNS
# --------------------------------------------------

columns_to_remove = [
    "Unnamed: 0.1",
    "Unnamed: 0",
    "User",
    "Retweets",
    "Likes"
]

df.drop(
    columns=[col for col in columns_to_remove if col in df.columns],
    inplace=True
)


# --------------------------------------------------
# 4. REMOVE DUPLICATES
# --------------------------------------------------

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print("Duplicate rows removed:", before - after)


# --------------------------------------------------
# 5. REMOVE MISSING VALUES
# --------------------------------------------------

df.dropna(
    subset=["Text", "Sentiment"],
    inplace=True
)


# --------------------------------------------------
# 6. SETUP TEXT CLEANING
# --------------------------------------------------

stop_words = set(stopwords.words("english"))

# IMPORTANT:
# Keep negation words because they can completely
# change the meaning of a sentence.

stop_words -= {
    "no",
    "not",
    "nor",
    "never"
}

lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# 7. TEXT CLEANING FUNCTION
# --------------------------------------------------

def clean_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Expand common contractions
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bdon't\b", "do not", text)
    text = re.sub(r"\bdoesn't\b", "does not", text)
    text = re.sub(r"\bdidn't\b", "did not", text)
    text = re.sub(r"\bisn't\b", "is not", text)
    text = re.sub(r"\baren't\b", "are not", text)
    text = re.sub(r"\bwasn't\b", "was not", text)
    text = re.sub(r"\bweren't\b", "were not", text)
    text = re.sub(r"\bshouldn't\b", "should not", text)
    text = re.sub(r"\bcouldn't\b", "could not", text)
    text = re.sub(r"\bwouldn't\b", "would not", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove # symbol but keep hashtag word
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove punctuation, numbers and special characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

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
# 8. APPLY TEXT CLEANING
# --------------------------------------------------

df["Text"] = df["Text"].apply(clean_text)


# --------------------------------------------------
# 9. CLEAN SENTIMENT LABELS
# --------------------------------------------------

df["Sentiment"] = (
    df["Sentiment"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# --------------------------------------------------
# 10. CONVERT EMOTIONS INTO 3 CLASSES
# --------------------------------------------------

positive_emotions = {
    "positive",
    "joy",
    "happy",
    "happiness",
    "excitement",
    "contentment",
    "gratitude",
    "relief",
    "acceptance",
    "determination",
    "serenity",
    "optimism",
    "hope",
    "pride",
    "satisfaction",
    "amusement",
    "admiration",
    "enthusiasm",
    "euphoria",
    "inspiration",
    "love"
}


negative_emotions = {
    "negative",
    "anger",
    "sadness",
    "fear",
    "disgust",
    "anxiety",
    "despair",
    "frustration",
    "hate",
    "boredom",
    "grief",
    "guilt",
    "shame",
    "loneliness",
    "numbness",
    "melancholy",
    "disappointment",
    "stress",
    "worry",
    "regret",
    "jealousy"
}


neutral_emotions = {
    "neutral",
    "indifference",
    "confusion",
    "ambivalence",
    "curiosity",
    "nostalgia",
    "surprise"
}


def convert_sentiment(emotion):

    if emotion in positive_emotions:
        return "positive"

    elif emotion in negative_emotions:
        return "negative"

    elif emotion in neutral_emotions:
        return "neutral"

    else:
        return "neutral"


df["Sentiment"] = df["Sentiment"].apply(
    convert_sentiment
)


# --------------------------------------------------
# 11. REMOVE EMPTY TEXT
# --------------------------------------------------

df = df[
    df["Text"].str.strip() != ""
]


# --------------------------------------------------
# 12. SAVE CLEANED DATA
# --------------------------------------------------

df.to_csv(
    "../data/clean_Sentiment.csv",
    index=False
)


# --------------------------------------------------
# 13. DISPLAY RESULTS
# --------------------------------------------------

print("\nCleaning completed!")

print("Final shape:", df.shape)

print("\nMissing values:")
print(
    df[["Text", "Sentiment"]]
    .isnull()
    .sum()
)

print("\nSentiment distribution:")
print(
    df["Sentiment"]
    .value_counts()
)

print("\nFirst 5 cleaned rows:")
print(
    df[["Text", "Sentiment"]].head()
)

print("\nSaved as:")
print("../data/clean_Sentiment.csv")