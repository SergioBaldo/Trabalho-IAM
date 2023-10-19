# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
from src.genetic_algorithm.genetic_search import GeneticSearch
from src.wrapper.wrapper import Wrapper
import xlsxwriter

# Load environment variables
load_dotenv(find_dotenv(), override=True)
DATA_PATH = os.environ.get("DATA_PATH")
SEED = int(os.environ.get("SEED"))
RESULTS_PATH = os.environ.get("RESULTS_PATH")
INDIVIDUAL_SIZE = int(os.environ.get("INDIVIDUAL_SIZE"))


def run_best_individual(clf, best_individual_ga, df_train, df_test, label_encoder):
    """
    Run a Random Forest model on the best individual selected by Genetic Algorithm.

    Args:
        clf (RandomForestClassifier): Random Forest classifier.
        best_individual_ga (list): List of feature names selected by Genetic Algorithm.
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Test data.
        label_encoder (LabelEncoder): The fitted LabelEncoder object.

    Returns:
        accuracy (float): Balanced accuracy score.
        f1_score_weighted (float): Weighted F1 score.
        conf_matrix (array): Confusion matrix.
    """
    random_forest = clf
    X_train = df_train[best_individual_ga]
    y_train = df_train.iloc[:, -1]

    X_test = df_test[best_individual_ga]
    y_test = df_test.iloc[:, -1]

    # Transform target labels using the provided LabelEncoder
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Train the Random Forest model
    random_forest.fit(X_train, y_train_encoded)

    # Make predictions on the test data
    y_pred_encoded = random_forest.predict(X_test)

    # Calculate the evaluation metrics
    accuracy = balanced_accuracy_score(y_test_encoded, y_pred_encoded)
    f1_score_weighted = f1_score(y_test_encoded, y_pred_encoded, average="weighted")

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)

    return accuracy, f1_score_weighted, conf_matrix


def run_model_with_all_features(clf, df_train, df_test, label_encoder):
    """
    Run a Random Forest model with all features.

    Args:
        clf (RandomForestClassifier): Random Forest classifier.
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Test data.
        label_encoder (LabelEncoder): The fitted LabelEncoder object.

    Returns:
        accuracy (float): Balanced accuracy score.
        f1_score_weighted (float): Weighted F1 score.
        conf_matrix (array): Confusion matrix.
    """
    random_forest = clf

    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # Transform target labels using the provided LabelEncoder
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Train the Random Forest model
    random_forest.fit(X_train, y_train_encoded)

    # Make predictions on the test data
    y_pred_encoded = random_forest.predict(X_test)

    # Calculate the evaluation metrics
    accuracy = balanced_accuracy_score(y_test_encoded, y_pred_encoded)
    f1_score_weighted = f1_score(y_test_encoded, y_pred_encoded, average="weighted")

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)

    return accuracy, f1_score_weighted, conf_matrix


def run_model_with_features_selected_by_random_forest(clf, df_train, df_test, label_encoder):
    """
    Run a Random Forest model with features selected based on Random Forest feature importances.

    Args:
        clf (RandomForestClassifier): Random Forest classifier.
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Test data.
        label_encoder (LabelEncoder): The fitted LabelEncoder object.

    Returns:
        accuracy (float): Balanced accuracy score.
        f1_score_weighted (float): Weighted F1 score.
        conf_matrix (array): Confusion matrix.
    """
    random_forest = clf

    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # Transform target labels using the provided LabelEncoder
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    random_forest.fit(X_train, y_train_encoded)

    # Get feature importances
    feature_importances = random_forest.feature_importances_

    # Get the indices of the top INDIVIDUAL_SIZE most important features
    top_features = np.argsort(feature_importances)[::-1][:INDIVIDUAL_SIZE]

    selected_features = df_train.columns[top_features]

    random_forest = clf
    X_train = df_train[selected_features]
    X_test = df_test[selected_features]

    # Train the Random Forest model
    random_forest.fit(X_train, y_train_encoded)

    # Make predictions on the test data
    y_pred_encoded = random_forest.predict(X_test)

    # Calculate the evaluation metrics
    accuracy = balanced_accuracy_score(y_test_encoded, y_pred_encoded)
    f1_score_weighted = f1_score(y_test_encoded, y_pred_encoded, average="weighted")

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)

    return accuracy, f1_score_weighted, conf_matrix


def save_results_to_xlsx(results, confusion_matrices, class_labels, file_path):
    """
    Save results and confusion matrices to an XLSX file with class labels in the confusion matrix.

    Args:
        results (dict): A dictionary containing the results.
        confusion_matrices (list of arrays): List of confusion matrices.
        class_labels (list): List of class labels.
        file_path (str): The path to the XLSX file where results will be saved.

    Returns:
        None
    """
    # Create a Pandas Excel writer
    writer = pd.ExcelWriter(file_path, engine="xlsxwriter")

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results)

    # Write the results DataFrame to the Excel file
    results_df.to_excel(writer, sheet_name="Results", index=False)

    # Create a sheet for each confusion matrix with truncated names
    for i, (algorithm_name, cm) in enumerate(zip(results["Model"], confusion_matrices)):
        # Truncate the name to fit within the 31-character limit
        truncated_name = algorithm_name[:40]

        # Create a DataFrame for each confusion matrix
        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

        # Write each confusion matrix DataFrame to a separate tab with the truncated name
        cm_df.to_excel(writer, sheet_name=truncated_name, index=True)

    # Save the Excel file
    writer._save()


def run_models(clf):
    """
    Run different models with various feature selection methods.

    Args:
        clf (RandomForestClassifier): Random Forest classifier.

    Returns:
        None
    """
    path_result_GA = RESULTS_PATH + f"/seed_{SEED}_IndividualSize_{INDIVIDUAL_SIZE}"

    df_train = pd.read_csv(f"{DATA_PATH}/processed/data_train.csv", index_col=None)
    df_test = pd.read_csv(f"{DATA_PATH}/processed/data_test.csv", index_col=None)

    try:
        with open(path_result_GA + "/BestIndividual.txt", "r") as file:
            # Read the contents of the file
            best_individual_ga = file.read().splitlines()
    except FileNotFoundError:
        print(f"The file does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df_train.iloc[:, -1])

    (
        accuracy_all_features,
        f1_score_all_features,
        conf_matrix_all_features,
    ) = run_model_with_all_features(clf, df_train, df_test, label_encoder)
    (
        accuracy_best_individual,
        f1_score_best_individual,
        conf_matrix_best_individual,
    ) = run_best_individual(clf, best_individual_ga, df_train, df_test, label_encoder)
    (
        accuracy_selected_features,
        f1_score_selected_features,
        conf_matrix_selected_features,
    ) = run_model_with_features_selected_by_random_forest(clf, df_train, df_test, label_encoder)

    # Define the path for saving results and confusion matrices
    results_file_path = path_result_GA + "/results.xlsx"

    results = {
        "Model": [
            "RF with All Features",
            "Best Individual (GA)",
            "RF with Top Features",
        ],
        "Balanced Accuracy": [
            accuracy_all_features,
            accuracy_best_individual,
            accuracy_selected_features,
        ],
        "Weighted F1 Score": [
            f1_score_all_features,
            f1_score_best_individual,
            f1_score_selected_features,
        ],
    }

    # Define class labels from the LabelEncoder
    class_labels = label_encoder.classes_

    confusion_matrices = [
        conf_matrix_all_features,
        conf_matrix_best_individual,
        conf_matrix_selected_features,
    ]

    # Save the results and confusion matrices to an XLSX file with class labels
    save_results_to_xlsx(results, confusion_matrices, class_labels, results_file_path)
