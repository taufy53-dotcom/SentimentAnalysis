# ==========================================================
# PYTORCH LSTM SENTIMENT ANALYSIS
# ==========================================================

import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from collections import Counter


# ==========================================================
# 1. REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================================
# 2. DEVICE
# ==========================================================

device = torch.device("cpu")

print("\nUsing device:", device)


# ==========================================================
# 3. CLOSE OLD PLOTS
# ==========================================================

# This makes sure figures created by THIS run
# do not stack on top of each other.

plt.close("all")


# ==========================================================
# 4. LOAD DATA
# ==========================================================

df = pd.read_csv("../data/clean_Sentiment_rewritten.csv")

print("\nDataset shape:", df.shape)

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())


# ==========================================================
# 5. CLEAN DATA
# ==========================================================

df["Text"] = df["Text"].fillna("").astype(str)

df = df[df["Text"].str.strip() != ""]

df = df.reset_index(drop=True)

print("\nDataset after removing empty text:", df.shape)


# ==========================================================
# 6. INPUT AND TARGET
# ==========================================================

X = df["Text"].tolist()
y = df["Sentiment"].tolist()


# ==========================================================
# 7. ENCODE LABELS
# ==========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\n==============================")
print("SENTIMENT LABELS")
print("==============================")

for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)


# ==========================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

# First:
# 80% training
# 20% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=SEED,
    stratify=y_encoded
)


# Then split temporary 50/50:
# 10% validation
# 10% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# ==========================================================
# 9. BUILD VOCABULARY
# ==========================================================

MAX_WORDS = 10000

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

counter = Counter()

for text in X_train:

    words = text.lower().split()

    counter.update(words)


most_common_words = counter.most_common(
    MAX_WORDS - 2
)


word_to_index = {

    PAD_TOKEN: 0,

    UNK_TOKEN: 1

}


for word, count in most_common_words:

    word_to_index[word] = len(word_to_index)


VOCAB_SIZE = len(word_to_index)

print("\nVocabulary size:", VOCAB_SIZE)


# ==========================================================
# 10. TEXT → SEQUENCE
# ==========================================================

MAX_LENGTH = 100


def text_to_sequence(text):

    words = text.lower().split()

    sequence = []

    for word in words:

        if word in word_to_index:

            sequence.append(
                word_to_index[word]
            )

        else:

            sequence.append(
                word_to_index[UNK_TOKEN]
            )


    # Truncate

    sequence = sequence[:MAX_LENGTH]


    # Padding

    while len(sequence) < MAX_LENGTH:

        sequence.append(
            word_to_index[PAD_TOKEN]
        )


    return sequence


# ==========================================================
# 11. CONVERT TEXT DATA
# ==========================================================

X_train_sequences = np.array(
    [
        text_to_sequence(text)
        for text in X_train
    ],
    dtype=np.int64
)


X_val_sequences = np.array(
    [
        text_to_sequence(text)
        for text in X_val
    ],
    dtype=np.int64
)


X_test_sequences = np.array(
    [
        text_to_sequence(text)
        for text in X_test
    ],
    dtype=np.int64
)


print("\nTraining sequence shape:",
      X_train_sequences.shape)

print("Validation sequence shape:",
      X_val_sequences.shape)

print("Testing sequence shape:",
      X_test_sequences.shape)


# ==========================================================
# 12. PYTORCH DATASET
# ==========================================================

class SentimentDataset(Dataset):

    def __init__(self, texts, labels):

        self.texts = torch.tensor(
            texts,
            dtype=torch.long
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, index):

        return (
            self.texts[index],
            self.labels[index]
        )


# ==========================================================
# 13. CREATE DATASETS
# ==========================================================

train_dataset = SentimentDataset(
    X_train_sequences,
    y_train
)


val_dataset = SentimentDataset(
    X_val_sequences,
    y_val
)


test_dataset = SentimentDataset(
    X_test_sequences,
    y_test
)


# ==========================================================
# 14. DATA LOADERS
# ==========================================================

BATCH_SIZE = 32


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================================
# 15. LSTM MODEL
# ==========================================================

class SentimentLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim
    ):

        super().__init__()


        # Word embeddings

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )


        # Bidirectional LSTM

        self.lstm = nn.LSTM(

            input_size=embedding_dim,

            hidden_size=hidden_dim,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.3
        )


        # Dropout

        self.dropout = nn.Dropout(0.5)


        # Because Bidirectional=True:
        #
        # hidden_dim * 2

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_dim
        )


    def forward(self, text):

        # text:
        # [batch, sequence_length]

        embedded = self.embedding(text)


        # embedded:
        # [batch, sequence_length, embedding_dim]

        output, (hidden, cell) = self.lstm(
            embedded
        )


        # hidden shape:
        #
        # [num_layers * 2,
        #  batch,
        #  hidden_dim]


        # Last forward hidden state

        forward_hidden = hidden[-2]


        # Last backward hidden state

        backward_hidden = hidden[-1]


        # Combine forward + backward

        hidden = torch.cat(
            (
                forward_hidden,
                backward_hidden
            ),
            dim=1
        )


        hidden = self.dropout(hidden)


        output = self.fc(hidden)


        return output


# ==========================================================
# 16. MODEL PARAMETERS
# ==========================================================

EMBEDDING_DIM = 128

HIDDEN_DIM = 128

OUTPUT_DIM = len(
    label_encoder.classes_
)


# ==========================================================
# 17. CREATE MODEL
# ==========================================================

model = SentimentLSTM(

    vocab_size=VOCAB_SIZE,

    embedding_dim=EMBEDDING_DIM,

    hidden_dim=HIDDEN_DIM,

    output_dim=OUTPUT_DIM
)


model = model.to(device)


print("\n==============================")
print("PYTORCH LSTM MODEL")
print("==============================")

print(model)


# ==========================================================
# 18. CLASS DISTRIBUTION
# ==========================================================

print("\n==============================")
print("TRAINING CLASS DISTRIBUTION")
print("==============================")


train_class_counts = np.bincount(
    y_train,
    minlength=OUTPUT_DIM
)


for i, count in enumerate(train_class_counts):

    print(
        f"{label_encoder.classes_[i]}: {count}"
    )


# ==========================================================
# 19. CLASS WEIGHTS
# ==========================================================

total_samples = len(y_train)

class_weights = []

for count in train_class_counts:

    weight = total_samples / (
        OUTPUT_DIM * count
    )

    class_weights.append(weight)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


print("\nClass weights:")

for i, weight in enumerate(class_weights):

    print(
        f"{label_encoder.classes_[i]}: "
        f"{weight.item():.4f}"
    )


# ==========================================================
# 20. LOSS FUNCTION
# ==========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==========================================================
# 21. OPTIMIZER
# ==========================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001,

    weight_decay=1e-5
)


# ==========================================================
# 22. TRAINING SETTINGS
# ==========================================================

EPOCHS = 20

best_val_loss = float("inf")

patience = 4

patience_counter = 0


train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []


# ==========================================================
# 23. TRAINING
# ==========================================================

print("\n==============================")
print("TRAINING PYTORCH LSTM")
print("==============================")


for epoch in range(EPOCHS):


    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    model.train()


    total_train_loss = 0

    train_correct = 0

    train_total = 0


    for texts, labels in train_loader:


        texts = texts.to(device)

        labels = labels.to(device)


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(texts)


        # Loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Prevent exploding gradients

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # Update weights

        optimizer.step()


        total_train_loss += loss.item()


        # Predictions

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_correct += (
            predictions == labels
        ).sum().item()


        train_total += labels.size(0)


    train_loss = (
        total_train_loss /
        len(train_loader)
    )


    train_accuracy = (
        train_correct /
        train_total
    )


    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    model.eval()


    total_val_loss = 0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for texts, labels in val_loader:


            texts = texts.to(device)

            labels = labels.to(device)


            outputs = model(texts)


            loss = criterion(
                outputs,
                labels
            )


            total_val_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += labels.size(0)


    val_loss = (
        total_val_loss /
        len(val_loader)
    )


    val_accuracy = (
        val_correct /
        val_total
    )


    # ------------------------------------------------------
    # STORE METRICS
    # ------------------------------------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # ------------------------------------------------------
    # PRINT
    # ------------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )


    # ------------------------------------------------------
    # EARLY STOPPING
    # ------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        patience_counter = 0


        # Save best model

        torch.save(
            model.state_dict(),
            "best_lstm_model.pth"
        )


        print("✓ Best model saved.")


    else:

        patience_counter += 1


        print(
            f"No improvement "
            f"({patience_counter}/{patience})"
        )


        if patience_counter >= patience:

            print(
                "\nEarly stopping."
            )

            break


# ==========================================================
# 24. LOAD BEST MODEL
# ==========================================================

print("\n==============================")
print("LOADING BEST MODEL")
print("==============================")


model.load_state_dict(
    torch.load(
        "best_lstm_model.pth",
        map_location=device
    )
)


model.eval()


# ==========================================================
# 25. TEST MODEL
# ==========================================================

all_predictions = []

all_labels = []

all_probabilities = []


with torch.no_grad():

    for texts, labels in test_loader:


        texts = texts.to(device)

        labels = labels.to(device)


        outputs = model(texts)


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_labels.extend(
            labels.cpu().numpy()
        )


        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


y_pred = np.array(
    all_predictions
)


y_true = np.array(
    all_labels
)


# ==========================================================
# 26. MODEL PERFORMANCE
# ==========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\n==============================")
print("FINAL MODEL PERFORMANCE")
print("==============================")


print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print("\nClassification Report:")


print(
    classification_report(

        y_true,

        y_pred,

        labels=np.arange(
            OUTPUT_DIM
        ),

        target_names=label_encoder.classes_,

        zero_division=0
    )
)


# ==========================================================
# 27. FRESH CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=np.arange(
        OUTPUT_DIM
    )
)


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")


print(cm)


# Close any previous figures

plt.close("all")


# Create NEW confusion matrix

fig, ax = plt.subplots(
    figsize=(7, 5)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=label_encoder.classes_,

    yticklabels=label_encoder.classes_,

    ax=ax,

    cbar=True
)


ax.set_xlabel(
    "Predicted Sentiment"
)


ax.set_ylabel(
    "Actual Sentiment"
)


ax.set_title(
    "PyTorch LSTM - Confusion Matrix"
)


plt.tight_layout()

plt.show()

plt.close(fig)


# ==========================================================
# 28. TRAINING ACCURACY GRAPH
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    range(1, len(train_accuracies) + 1),

    train_accuracies,

    label="Training Accuracy"
)


ax.plot(
    range(1, len(val_accuracies) + 1),

    val_accuracies,

    label="Validation Accuracy"
)


ax.set_xlabel(
    "Epoch"
)


ax.set_ylabel(
    "Accuracy"
)


ax.set_title(
    "PyTorch LSTM Training vs Validation Accuracy"
)


ax.legend()


plt.tight_layout()

plt.show()

plt.close(fig)


# ==========================================================
# 29. TRAINING LOSS GRAPH
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    range(1, len(train_losses) + 1),

    train_losses,

    label="Training Loss"
)


ax.plot(
    range(1, len(val_losses) + 1),

    val_losses,

    label="Validation Loss"
)


ax.set_xlabel(
    "Epoch"
)


ax.set_ylabel(
    "Loss"
)


ax.set_title(
    "PyTorch LSTM Training vs Validation Loss"
)


ax.legend()


plt.tight_layout()

plt.show()

plt.close(fig)


# ==========================================================
# 30. TEST YOUR OWN TEXT
# ==========================================================

print("\n==============================")
print("TEST YOUR OWN TEXT")
print("==============================")


def clean_input_text(text):

    text = str(text).lower()

    # Remove URLs

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove mentions

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtag symbol

    text = re.sub(
        r"#",
        "",
        text
    )

    # Keep letters and spaces

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

    return text


while True:


    text = input(
        "\nEnter a sentence "
        "(or type 'exit'): "
    )


    if text.strip().lower() == "exit":

        print("\nExiting...")

        break


    if not text.strip():

        print(
            "Please enter some text."
        )

        continue


    # Clean input

    cleaned_text = clean_input_text(
        text
    )


    # Convert to sequence

    sequence = text_to_sequence(
        cleaned_text
    )


    # Convert to tensor

    tensor = torch.tensor(
        [sequence],
        dtype=torch.long
    ).to(device)


    # Prediction

    model.eval()


    with torch.no_grad():

        output = model(
            tensor
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


        prediction_index = torch.argmax(
            probabilities
        ).item()


    # Convert index → sentiment

    prediction = (
        label_encoder
        .inverse_transform(
            [prediction_index]
        )[0]
    )


    # Confidence

    confidence = (
        probabilities[
            prediction_index
        ].item()
        * 100
    )


    print(
        "\nPredicted Sentiment:",
        prediction.upper()
    )


    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )


    # ------------------------------------------------------
    # SHOW ALL PROBABILITIES
    # ------------------------------------------------------

    print("\nProbabilities:")


    for i, label in enumerate(
        label_encoder.classes_
    ):

        probability = (
            probabilities[i].item()
            * 100
        )


        print(
            f"{label.capitalize():10} "
            f"{probability:.2f}%"
        )