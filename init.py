import numpy as np
import pandas as pd
import torch
import nltk


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


def numerical_labels(pos):
    d = {}
    count = 0
    for i in pos:
        d[i] = count
        count += 1
    return d


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
    X = data["Word"]
    Y = data["POS_Tag"]
    print(set(Y.values))
    unique_labels = set(Y.values)
    d = numerical_labels(unique_labels)
    print(d)

    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(processed_data, labels)

    # Initialize and train the model
    model = CustomModel()
    train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    """
