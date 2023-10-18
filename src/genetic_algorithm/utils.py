import random


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
    num_elements_per_individual = num_features // individual_size + 1
    remaining_features = feature_names[:]

    # Create individual feature groups
    for i in range(individual_size):
        random.seed(42)
        if num_features > 0:
            # Determine the number of elements to select
            num_elements_to_select = min(num_features, num_elements_per_individual)
            # Randomly select a subset of features
            selected_subset = random.sample(remaining_features, num_elements_to_select)
            features_names_dict[i] = selected_subset
            num_features -= num_elements_to_select
            # print(num_features)

            # Remove selected elements from remaining_features
            remaining_features = [
                feature for feature in remaining_features if feature not in selected_subset
            ]

    return features_names_dict
