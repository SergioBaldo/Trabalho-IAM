import random
from src.models.RandomForestClassifierWrapper import RandomForestClassifierWrapper
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), override=True)
SEED = int(os.environ.get("SEED"))

random.seed(SEED)


def create_feature_groups(feature_names, individual_size):
    """
    Create feature groups by randomly selecting subsets of feature names.

    Args:
        feature_names (list): A list of feature names.
        individual_size (int): The number of feature groups to create.

    Returns:
        dict: A dictionary containing the selected feature groups.
    """

    # Check if "subtype" exists in feature_names and remove it
    if "subtype" in feature_names:
        feature_names.remove("subtype")

    # Initialize a dictionary to store the selected feature groups
    features_names_dict = {}

    num_features = len(feature_names)
    num_elements_per_individual = (num_features // individual_size) + 1
    remaining_features = feature_names[:]

    # Create individual feature groups
    for i in range(individual_size):
        if num_features > 0:
            # Determine the number of elements to select
            num_elements_to_select = min(num_features, num_elements_per_individual)
            # Randomly select a subset of features
            selected_subset = random.sample(remaining_features, num_elements_to_select)
            features_names_dict[i] = selected_subset
            num_features -= num_elements_to_select

            # Remove selected elements from remaining_features
            remaining_features = [
                feature for feature in remaining_features if feature not in selected_subset
            ]

    return features_names_dict


def run_best_individual(data_path, best_individual, generation):
    """
    Test the best individual with the original train and test datasets and display the results.

    Args:
        data_path (str): The path to the data file.
        best_individual (list): The best features selected by the individual.
        generation (int): The generation number.

    Prints:
        - Random Forest Model Balanced Accuracy and Weighted Averaged F1 Score with the best features.
        - Balanced Accuracy
        - Weighted Averaged F1 Score
        - Confusion Matrix
    """
    # Test the best individual
    rf_wrapper = RandomForestClassifierWrapper(
        data_file=data_path, target_column="subtype", best_features=best_individual
    )
    accuracy, f1_score, confusion_matrix = rf_wrapper.train_and_evaluate()

    print(
        f"- Random Forest Model: Balanced Accuracy and Weighted Averaged F1 Score with the best features of generation {generation}"
    )
    print(f"- Balanced Accuracy: {accuracy:.2f}, Weighted Averaged F1 Score: {f1_score:.2f}")
    print("- Confusion Matrix")
    print(confusion_matrix)
