import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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
    np.random.shuffle(feature_names)

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


def fitness_function(individual, df_train, df_validation, fitness_has_table):
    pass


def nominal_mutation(children, name_of_features, mutation_prob=0.02):
    """
    Apply nominal mutation to the children by randomly changing feature values.

    Parameters:
        children (list): List of feature values for the children.
        name_of_features (list): List of available feature values to choose from.
        mutation_prob (float): Probability of mutation for each gene.

    Returns:
        list: The mutated children with updated feature values.
    """
    mutated_children = children.copy()  # Create a copy to avoid modifying the original list

    for gene in range(len(mutated_children)):
        if np.random.rand() < mutation_prob:
            # Remove the current gene value from available options
            name_of_features.remove(mutated_children[gene])

            # Randomly select a new feature value from the remaining options
            new_value = np.random.choice(name_of_features, 1, replace=False)[0]

            # Update the child's gene with the new value
            mutated_children[gene] = new_value

    return mutated_children
