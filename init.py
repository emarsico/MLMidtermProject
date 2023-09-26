# 1. Import necessary libraries
import numpy as np
import pandas as pd
import torch
import nltk


# 2. Load data
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


# 3. Preprocess data
def preprocess_data(data):
    # TODO: Implement data preprocessing steps
    pass


# 4. Split data into training and testing sets
def split_data(data, labels):
    # TODO: Implement data splitting
    pass


# 5. Define the ML model
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


# 6. Train the model
def train_model(model, X_train, y_train):
    # TODO: Implement the model training
    pass


# 7. Evaluate the model
def evaluate_model(model, X_test, y_test):
    # TODO: Implement model evaluation
    pass


# 8. Main execution
if __name__ == "__main__":
    # Load data
    lines = load_data("train.txt")
    print(lines.head)

    # Preprocess data
    """
    processed_data = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(processed_data, labels)

    # Initialize and train the model
    model = CustomModel()
    train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    """
