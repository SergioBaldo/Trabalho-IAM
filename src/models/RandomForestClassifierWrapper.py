import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), override=True)
SEED = int(os.environ.get("SEED"))

np.random.seed(SEED)


class RandomForestClassifierWrapper:
    def __init__(
        self, data_file, target_column, best_features, n_estimators=100, random_state=SEED
    ):
        """
        Initialize the RandomForestClassifierWrapper.

        Args:
            data_file (str): Path to the CSV file containing the dataset.
            target_column (str): Name of the target column in the dataset.
            n_estimators (int): Number of trees in the Random Forest (default is 100).
            random_state (int): Random seed for reproducibility (default is 42).
        """
        self.data_file = data_file
        self.target_column = target_column
        self.best_features = best_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train_and_evaluate(self):
        """
        Load the dataset, train a Random Forest model, and return the accuracy.

        Args:
            test_size (float): Proportion of the dataset to use for testing (default is 0.2).

        Returns:
            float: Accuracy of the Random Forest model on the test data.
        """
        # Load the dataset
        data_train = pd.read_csv(self.data_file + "/processed/data_train.csv")
        data_test = pd.read_csv(self.data_file + "/processed/data_test.csv")

        # Extract selected features from the training and test datasets
        X_train = data_train[self.best_features].values
        X_test = data_test[self.best_features].values

        y_train = data_train[self.target_column]
        y_test = data_test[self.target_column]

        # Encode target labels using LabelEncoder
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Train the Random Forest model
        self.model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = self.model.predict(X_test)

        # Calculate and return the accuracy
        accuracy = balanced_accuracy_score(y_test, y_pred)
        f1_score_weighted = f1_score(y_test, y_pred, average="weighted")

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        return (accuracy, f1_score_weighted, conf_matrix)
