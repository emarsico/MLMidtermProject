from math import floor
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import nltk
from nltk.classify import NaiveBayesClassifier

# Logistic Regression Imports
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from nltk.classify import accuracy


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


# Not done
def distances_from_end_of_sentence(text):
    result = []
    for sentence in np.array_split(text, np.where(text == ".")[0] + 1):
        for i, word in enumerate(sentence):
            distance_from_end = len(sentence) - i - 1
            result.extend([distance_from_end])
    return result


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = ["a", "e", "i", "o", "u"]
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return count if count != 0 else 1


# Split data into training and testing sets
def split_data(data, labels):
    # TODO: Implement data splitting
    pass

# Main execution
if __name__ == "__main__":
    # Load data
    data = load_data("train.txt")
    # print(set(Y.values))
    # unique_labels = set(Y.values)
    # d = numerical_labels(unique_labels)
    # print(d)
    data["Capitalized"] = data["Word"].apply(is_capitalized)
    data["Length"] = data["Word"].apply(len)
    data["Prefix"] = data["Word"].apply(prefix)
    data["Suffix"] = data["Word"].apply(suffix)
    data["Syllables"] = data["Word"].apply(syllable_count)
    distances = distance_from_period(data["Word"].values)
    distance_from_end = distances_from_end_of_sentence(data["Word"].values)
    print(distance_from_end)
    for index, row in data.iterrows():
        data.at[index, "Position"] = floor(distances[index])
    for index, row in data.iterrows():
        data.at[index, "Distance_From_End"] = floor(distance_from_end[index])
    X = data[
        [
            "Word",
            "Capitalized",
            "Length",
            "Prefix",
            "Suffix",
            "Position",
            "Distance_From_End",
        ]
    ]
    # X = data[["Word"]]
    Y = data["POS_Tag"]
    features = [
        ({col: row[col] for col in X.columns}, label)
        for index, (index, row), label in zip(X.iterrows(), data.iterrows(), Y)
    ]
    random.shuffle(features)

    train_set, test_set = train_test_split(features, test_size=0.2, random_state=42)

    # Train
    classifier = NaiveBayesClassifier.train(train_set)

    # Evaluate
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    print(f"Classifier Accuracy: {accuracy}")