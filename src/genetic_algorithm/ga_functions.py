import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from dotenv import load_dotenv, find_dotenv
import os
from sklearn.model_selection import StratifiedKFold


load_dotenv(find_dotenv(), override=True)
SEED = int(os.environ.get("SEED"))
np.random.seed(SEED)


def create_individual(individual_size, feature_names):
    """
    Create an individual for a genetic algorithm with a specified size.

    Parameters:
        individual_size (int): The number of features in the individual.
        name_of_features (list): List of available feature names to choose from.

    Returns:
        list: An individual with randomly selected feature names.
    """
    # Shuffle the list of feature names to create a random order
    np.random.shuffle(
        feature_names,
    )

    # Create the individual by selecting the first 'individual_size' names
    individual = feature_names[:individual_size]

    return individual


def roulette_selection(fitness, population, k_tour=2):
    """
    Perform roulette wheel selection to choose two parents from the population based on their fitness values.

    Parameters:
        fitness (numpy.ndarray): Array containing the fitness values of individuals in the population.
        population (numpy.ndarray): Array containing the population of individuals.
        k_tour (int): Number of individuals to select in each tournament.

    Returns:
        tuple: A tuple containing two selected parents.
    """
    # Select two parents using roulette wheel selection
    id_parents = np.random.choice(
        population.shape[0], k_tour, replace=False, p=fitness / sum(fitness)
    )
    parent1 = population[id_parents[0]]
    parent2 = population[id_parents[1]]
    return parent1, parent2


def tournament_selection(fitness, population, k_tour=2):
    """
    Perform tournament selection to choose two parents from the population based on their fitness values.

    Parameters:
        fitness (numpy.ndarray): Array containing the fitness values of individuals in the population.
        population (numpy.ndarray): Array containing the population of individuals.
        k_tour (int): Number of individuals to select in each tournament.

    Returns:
        tuple: A tuple containing two selected parents.
    """
    # Select random individuals for tournaments
    id_parent1 = np.random.choice(population.shape[0], k_tour, replace=True)
    id_parent2 = np.random.choice(population.shape[0], k_tour, replace=True)

    # Select the best individuals from each tournament
    idx1 = np.argmax(fitness[id_parent1])
    idx2 = np.argmax(fitness[id_parent2])

    # Choose the parents based on tournament results
    parent1 = population[id_parent1[idx1]]
    parent2 = population[id_parent2[idx2]]

    return parent1, parent2


def two_point_crossover(parent1, parent2, cprob):
    """
    Perform two-point crossover between two parents with a given crossover probability.

    Parameters:
        parent1 (numpy.ndarray): First parent for crossover.
        parent2 (numpy.ndarray): Second parent for crossover.
        cprob (float): Crossover probability.

    Returns:
        tuple: A tuple containing two children resulting from crossover.
    """
    if np.random.rand() < cprob:
        # Perform two-point crossover
        points_crossover = np.random.choice(np.arange(1, len(parent1)), 2, replace=True)
        points_crossover.sort()

        children1 = np.concatenate(
            (
                parent1[: points_crossover[0]],
                parent2[points_crossover[0] : points_crossover[1]],
                parent1[points_crossover[1] :],
            )
        )

        children2 = np.concatenate(
            (
                parent2[: points_crossover[0]],
                parent1[points_crossover[0] : points_crossover[1]],
                parent2[points_crossover[1] :],
            )
        )
    else:
        # If no crossover, children are identical to parents
        children1 = parent1.copy()
        children2 = parent2.copy()

    return children1, children2


def fitness_function(clf, X, y, individual, fitness_hash_table):
    """
    Calculate fitness for an individual.

    This function measures the performance of a classifier on a dataset using a specific set of features.

    Parameters:
    - clf (object): A machine learning classifier.
    - X (pandas.DataFrame): The dataset with features.
    - y (pandas.Series): The target labels.
    - individual (list): A list of feature names to be selected.
    - fitness_cache (dict): A cache to store fitness values for reuse.

    Returns:
    - tuple: A tuple containing accuracy and weighted F1-score.
    """

    # Convert the individual list into a hash key for caching
    hash_key = "".join(np.array(np.array(individual, dtype="O"), dtype="O"))

    # Check if the fitness value is already calculated and cached
    if hash_key in fitness_hash_table:
        return fitness_hash_table[hash_key]

    # Extract the selected features from the dataset
    X_selected = X[individual]

    # Stratified k-fold cross-validation
    skfolds = StratifiedKFold(n_splits=10)

    accuracy_scores = []
    weighted_f1_scores = []

    for train_index, test_index in skfolds.split(X_selected, y):
        clf_cv = clf
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Encode target labels using LabelEncoder
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Train the classifier
        clf_cv.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = clf_cv.predict(X_test)

        # Calculate accuracy and weighted F1-score
        accuracy = balanced_accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        accuracy_scores.append(accuracy)
        weighted_f1_scores.append(weighted_f1)

    mean_accuracy = np.mean(accuracy_scores)
    mean_weighted_f1 = np.mean(weighted_f1_scores)

    # Cache the fitness value for reuse
    fitness_hash_table[hash_key] = (mean_accuracy, mean_weighted_f1)

    return mean_accuracy, mean_weighted_f1


def nominal_mutation(children, features_groups, mutation_prob=0.02):
    """
    Apply nominal mutation to the children by randomly changing feature values.

    Parameters:
        children (list): List of feature values for the children.
        features_groups (dict): A dictionary of feature groups, where each key corresponds to a gene and
                               the value is a list of possible feature values for that gene.
        mutation_prob (float): Probability of mutation for each gene.

    Returns:
        list: The mutated children with updated feature values.
    """
    mutated_children = children.copy()  # Create a copy to avoid modifying the original list

    for gene in range(len(mutated_children)):
        if np.random.rand() < mutation_prob:
            # Randomly select a new feature value from the options
            new_value = np.random.choice(features_groups[gene], 1, replace=False)[0]

            # Update the child's gene with the new value
            mutated_children[gene] = new_value

    return mutated_children
