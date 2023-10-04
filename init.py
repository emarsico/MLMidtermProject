from math import floor
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
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


def numerical_values(pos):
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


def word_encoding(word):
    return int("".join(str(ord(char)) for char in word))


# Split data into training and testing sets
def split_data(data, labels):
    # TODO: Implement data splitting
    pass


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.softmax(out, dim=1)
        # potentially softmax
        # NOt done
        # out = self.sigmoid(out)
        return out


# Main execution
if __name__ == "__main__":
    # Load data
    data = load_data("train.txt")
    # print(set(Y.values))
    # unique_labels = set(Y.values)
    # d = numerical_values(unique_labels)
    # print(d)
    """
    data["Capitalized"] = data["Word"].apply(is_capitalized)
    data["Length"] = data["Word"].apply(len)
    data["Prefix"] = data["Word"].apply(prefix)
    data["Suffix"] = data["Word"].apply(suffix)
    data["Syllables"] = data["Word"].apply(syllable_count)
    distances = distance_from_period(data["Word"].values)
    distance_from_end = distances_from_end_of_sentence(data["Word"].values)
    sentance_length = [x + y for x, y in zip(distances, distance_from_end)]
    data["Previous"] = data["Word"].shift(1)
    data["Next"] = data["Word"].shift(-1)
    for index, row in data.iterrows():
        
        data.at[index, "Distance_From_End"] = floor(distance_from_end[index])
        data.at[index, "Position"] = floor(distances[index])
        if distances[index]:
            data.at[index, "Spot"] = sentance_length[index] / distances[index]
        else:
            data.at[index, "Spot"] = 0
        """

    """X = data[
        [
            "Word",
            "Capitalized",
            "Length",
            "Prefix",
            "Suffix",
            "Position",
            "Distance_From_End",
            "Previous",
            "Next",
        ]
    ]"""
    """"""
    X = data[["Word"]]

    # X = data[["Word"]]
    Y = data["POS_Tag"]
    features = [
        ({col: row[col] for col in X.columns}, label)
        for index, (index, row), label in zip(X.iterrows(), data.iterrows(), Y)
    ]

    # Split data into training and testing sets
    """
    train_set, test_set = train_test_split(features, test_size=0.2, random_state=42)

    # Train
    classifier = NaiveBayesClassifier.train(train_set)

    # Evaluate
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    print(f"Classifier Accuracy: {accuracy}")
    """

    """Logistical Regression"""
    unique_labels = set(Y.values)
    d = numerical_values(unique_labels)
    unique_words = set(data["Word"].values)
    d2 = numerical_values(unique_words)
    data["POS_Hashed"] = data["POS_Tag"].map(d)
    data["Numerical_Words"] = data["Word"].map(d2)
    # need to encode mroe
    Xl = torch.tensor(data["Numerical_Words"].values, dtype=torch.float32)
    Yl = torch.tensor(data["POS_Hashed"].values, dtype=torch.long)
    # print(Xl.shape)
    # check
    model = LogisticRegression(len(unique_words), len(unique_labels))

    cross = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(Xl)
        loss = cross(outputs, Yl)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        predicted = torch.argmax(model(X), dim=1)

    accuracy = (predicted == y).float().mean()
    print(f"Accuracy: {accuracy.item()*100:.2f}%")
