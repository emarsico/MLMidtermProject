import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Open training data and split into sentences
data = []
with open("train.txt", "r") as file:
    sentence = []
    for line in file:
        if line.strip():
            word, pos_tag, chunk_tag = line.strip().split()
            sentence.append([word, pos_tag])
        else:
            data.append(sentence)
            sentence = []

# Use only a subset of data for faster processing (e.g., 10%)
data = data[: int(len(data))]


def feature_extraction(sentence, i):
    word = sentence[i][0]
    last_word = sentence[i - 1][0] if i > 0 else "<START>"
    next_word = sentence[i + 1][0] if i < len(sentence) - 1 else "<END>"
    distance_from_end = len(sentence) - i - 1
    return {
        "word_length": len(word),  # Get length of each word
        "is_capitalized": word[0].isupper(),  # Get capitalization of each word
        "has_number": any(char.isdigit() for char in word),  # Get if any digits in word
        "prefix2": word[:3],  # Get next possible prefix
        "prefix3": word[:4],  # Get last possible prefix
        "suffix1": word[-1:],  # Get possible suffix
        "suffix2": word[-2:],  # Get next possible suffix
        "suffix3": word[-3:],  # Get last possible suffix
        "last_word": last_word,  # Get word that came before
        "next_word": next_word,  # Get word that comes next
    }


# Extract features
X, y = [], []
for sentence in data:
    for i in range(len(sentence)):
        X.append(feature_extraction(sentence, i))
        y.append(sentence[i][1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert features to matrix
dict_vectorizer = DictVectorizer(sparse=True)
X_train_transformed = dict_vectorizer.fit_transform(X_train)
X_test_transformed = dict_vectorizer.transform(X_test)

# Train SVM classifier with a simpler kernel and adjusted parameters
svm_classifier = SVC(
    kernel="linear", C=0.0, random_state=42, verbose=1, max_iter=5000, tol=0.01
)  # set max_iter and adjust tol
svm_classifier.fit(X_train_transformed, y_train)

# Predict
y_pred = svm_classifier.predict(X_test_transformed)

# Print results
print(classification_report(y_test, y_pred))
