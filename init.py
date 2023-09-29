from math import floor
import random
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.classify import NaiveBayesClassifier


# Load data
def load_data(file_name):
    words = []
    pos_tags = []
    entity_labels = []
    with open(file_name, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            word, pos, entity = parts
            words.append(word)
            pos_tags.append(pos)
    df = pd.DataFrame({"Word": words, "POS_Tag": pos_tags})
    return df


def word_features(row):
    return {
        "Word": row[
            0
        ],  # Accessing the first element, which corresponds to the 'Word' column
        "Capitalized": is_capitalized(row[2]),
        "Length": row[3],
        "Prefix": row[4],
        "Suffix": row[5],
        "Position": row[6],
    }


def numerical_labels(pos):
    d = {}
    count = 0
    for i in pos:
        d[i] = count
        count += 1
    return d


# Returns true if a word is capitalized, need to implement functionality for punctuations
def is_capitalized(word):
    if word[0] == word[0].upper():
        return 1
    return 0


def prefix(word):
    prefixes = [
        "multi",
        "over",
        "un",
        "dis",
        "in",
        "pre",
        "inter",
        "re",
        "co",
        "sub",
        "mis",
        "anti",
        "ex",
        "tele",
        "bi",
    ]
    for i in range(0, len(prefixes)):
        if word.startswith(prefixes[i]):
            return i
    # no prefix numerical value
    return len(prefixes) + 1


def suffix(word):
    suffixes = [
        "ful",
        "s",
        "able",
        "ize",
        "ing",
        "ment",
        "ed",
        "sion",
        "ity",
        "ness",
        "tion",
        "er",
        "less",
        "est",
        "ly",
        "es",
        "ible",
        "ise",
    ]
    for i in range(0, len(suffixes)):
        if word.endswith(suffixes[i]):
            return i
    # no suffix numerical value
    return len(suffixes) + 1


def distance_from_period(text):
    l = []
    count = 0
    for i in range(len(text)):
        if text[i] == ".":
            count = 0
        l.append(count)
        count = count + 1
    return l


# Split data into training and testing sets
def split_data(data, labels):
    # TODO: Implement data splitting
    pass


# Define the ML model
class CustomModel:
    def __init__(self):
        # TODO: Define model parameters
        pass

    def train(self, X_train, y_train):
        # TODO: Implement training procedure
        pass

    def predict(self, X_test):
        # TODO: Implement prediction procedure
        pass


# Train the model
def train_model(model, X_train, y_train):
    # TODO: Implement the model training
    pass


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    # TODO: Implement model evaluation
    pass


# Main execution
if __name__ == "__main__":
    # Load data
    data = load_data("train.txt")
    print(data.head)
    # print(set(Y.values))
    # unique_labels = set(Y.values)
    # d = numerical_labels(unique_labels)
    # print(d)
    data["Capitalized"] = data["Word"].apply(is_capitalized)
    data["Length"] = data["Word"].apply(len)
    data["Prefix"] = data["Word"].apply(prefix)
    data["Suffix"] = data["Word"].apply(suffix)
    distances = distance_from_period(data["Word"].values)
    for index, row in data.iterrows():
        data.at[index, "Position"] = floor(distances[index])

    print(data.columns)
    X = data[["Word", "Capitalized", "Length", "Prefix", "Suffix", "Position"]]
    Y = data["POS_Tag"]
    featuresets = [
        ({col: row[col] for col in X.columns}, label)
        for index, (index, row), label in zip(X.iterrows(), data.iterrows(), Y)
    ]
    random.shuffle(featuresets)

    # Split data into training and testing sets
    train_set, test_set = (
        featuresets[: int(len(featuresets) * 0.8)],
        featuresets[int(len(featuresets) * 0.8) :],
    )

    # Train
    classifier = NaiveBayesClassifier.train(train_set)

    # Evaluate
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    print(f"Classifier Accuracy: {accuracy}")

    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(processed_data, labels)

    # Initialize and train the model
    model = CustomModel()
    train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    """
