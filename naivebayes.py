from math import floor
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import nltk
from nltk.classify import NaiveBayesClassifier
import syllables

# Logistic Regression Imports
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from nltk.classify import accuracy


# Load data
def load_data(file_name):
    words = []
    pos_tags = []
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


# Returns true if a word is capitalized, need to implement functionality for punctuations
def is_capitalized(word):
    if word[0] == word[0].upper():
        return 1
    return 0


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


# Not done
def distances_from_end_of_sentence(text):
    result = []
    for sentence in np.array_split(text, np.where(text == ".")[0] + 1):
        for i, word in enumerate(sentence):
            distance_from_end = len(sentence) - i - 1
            result.extend([distance_from_end])
    return result


def word_frequency(words):
    d = {}
    for i in words:
        if i in d:
            d[i] += 1
        else:
            d[i] = 0
    return d


def in_parenthesis(text):
    p = []
    opened = 0
    for i in text:
        if i == "-LRB-":
            opened = 1
            p.append(2)
        elif i == "-RRB-":
            opened = 0
            p.append(2)
        else:
            p.append(opened)
    return p


# Main execution
if __name__ == "__main__":
    # Load data
    data = load_data("train.txt")

    distances = distance_from_period(data["Word"].values)
    distance_from_end = distances_from_end_of_sentence(data["Word"].values)
    frequency = word_frequency(data["Word"].values)
    parenthesis = in_parenthesis(data["Word"].values)

    data["Capitalized"] = data["Word"].apply(is_capitalized)
    data["Length"] = data["Word"].apply(len)
    data["Suffix"] = data["Word"].apply(suffix)
    data["Prefix"] = data["Word"].apply(lambda x: x[:3])
    data["Frequency"] = data["Word"].map(frequency)
    data["Number"] = data["Word"].apply(
        lambda word: any(char.isdigit() for char in word)
    )
    data["Next"] = data["Word"].shift(1)
    data["Up_Two"] = data["Word"].shift(2)
    data["Previous"] = data["Word"].shift(-1)
    data["Back_Two"] = data["Word"].shift(-2)
    data["After_Parenthesis"] = data.index.to_series().apply(lambda x: parenthesis[x])

    X = data[
        [
            "Word",
            "Capitalized",
            "Length",
            "Suffix",
            "Previous",
            "Next",
            "Frequency",
            "Prefix",
            "Number",
            "Up_Two",
            "Back_Two",
            "After_Parenthesis",
        ]
    ]
    Y = data["POS_Tag"]
    features = [
        ({col: row[col] for col in X.columns}, label)
        for index, (index, row), label in zip(X.iterrows(), data.iterrows(), Y)
    ]
    # Split data
    train_set, test_set = train_test_split(features, test_size=0.2, random_state=42)

    # Train
    classifier = NaiveBayesClassifier.train(train_set)

    # Evaluate
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    print(f"Classifier Accuracy: {accuracy}")
