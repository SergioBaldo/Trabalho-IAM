import os
import time
import pandas as pd
import numpy as np
import random
from tqdm import trange
from dotenv import load_dotenv, find_dotenv
from src.genetic_algorithm.ga_functions import (
    tournament_selection,
    nominal_mutation,
    two_point_crossover,
    fitness_function,
    create_individual,
)
from src.utils.utils_ga import create_feature_groups
from src.genetic_algorithm.save_ga_results import SaveResults

load_dotenv(find_dotenv(), override=True)
POPSIZE = int(os.environ.get("POPSIZE"))
N_GENERATION = int(os.environ.get("N_GENERATION"))
CROSSOVER_PROB = float(os.environ.get("CROSSOVER_PROB"))
MUTATION_PROB = float(os.environ.get("MUTATION_PROB"))
TOUR_SIZE = int(os.environ.get("TOUR_SIZE"))
INDIVIDUAL_SIZE = int(os.environ.get("INDIVIDUAL_SIZE"))
ELITISM = eval(os.environ.get("ELITISM"))
SEED = int(os.environ.get("SEED"))


np.random.seed(SEED)
random.seed(SEED)


class GeneticSearch:
    def __init__(
        self,
        popsize=POPSIZE,
        ngeneration=N_GENERATION,
        cprob=CROSSOVER_PROB,
        mprob=MUTATION_PROB,
        tour_size=TOUR_SIZE,
        individual_size=INDIVIDUAL_SIZE,
        elitism=ELITISM,
        selection=tournament_selection,
        mutation=nominal_mutation,
        crossover=two_point_crossover,
        fitness_function=fitness_function,
    ):
        """
        A class for performing genetic search for feature selection.

        This class initializes populations, calculates fitness, and evolves populations over generations to find the best feature subset for a given classifier and dataset.

        Args:
            popsize (int): Population size.
            ngeneration (int): Number of generations.
            cprob (float): Crossover probability.
            mprob (float): Mutation rate.
            tour_size (int): Tournament size for selection.
            individual_size (int): Size of each individual.
            elitism (bool): Whether to use elitism.
            selection (function): The selection function.
            mutation (function): The mutation function.
            crossover (function): The crossover function.
            fitness_function (function): The fitness evaluation function.
        """
        self.popsize = popsize
        self.ngeneration = ngeneration
        self.cprob = cprob
        self.mprob = mprob
        self.tour_size = tour_size
        self.individual_size = individual_size
        self.elitism = elitism
        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover
        self.fitness_function = fitness_function

    def search(self, clf, X, y):
        """
        Perform genetic search for feature selection.

        This method initializes populations, calculates fitness, and evolves populations over generations to find the best feature subset for a given classifier and dataset.

        Args:
            clf: The classifier for evaluating feature subsets.
            X (DataFrame): The input data.
            y (Series): The target labels.

        Returns:
            None
        """
        feature_names = list(X.columns)

        feature_groups = create_feature_groups(
            feature_names=feature_names, individual_size=self.individual_size
        )
        start_time_GA = time.time()

        # Generate a population with M individuals (initial population)
        population = np.zeros((self.popsize, self.individual_size), dtype="O")

        fitness_has_table = {}
        output = np.zeros((self.ngeneration, 3))
        output_score = np.zeros(self.ngeneration)
        bestFitness = float("-inf")

        # Generate individuals
        for i in range(self.popsize):
            population[i] = create_individual(
                individual_size=self.individual_size, feature_names=feature_names
            )

        for gen in range(self.ngeneration):
            print("")
            fitness = np.zeros(self.popsize)
            accuracy = np.zeros(self.popsize)

            for individual in trange(self.popsize, desc=f"Generation: {gen}"):
                accuracy[individual], fitness[individual] = self.fitness_function(
                    clf=clf,
                    X=X,
                    y=y,
                    individual=population[individual],
                    fitness_hash_table=fitness_has_table,
                )

            output_score[gen] = np.max(accuracy)
            output[gen, 0] = np.mean(fitness)
            output[gen, 1] = np.std(fitness)
            output[gen, 2] = np.max(fitness)

            bestFitness_Id = np.argmax(fitness)

            if bestFitness < np.max(fitness):
                bestInd = population[bestFitness_Id]
                bestFitness = np.max(fitness)

            print(f"- Best Fitness (F1 Score Weighted) generation {gen}: {bestFitness:.2f}")

            children = np.zeros((self.popsize, self.individual_size), dtype="O")
            for j in range(int(self.popsize / 2)):
                parent1, parent2 = self.selection(
                    fitness=fitness, population=population, k_tour=self.tour_size
                )

                child1, child2 = self.crossover(parent1=parent1, parent2=parent2, cprob=self.cprob)

                child1 = self.mutation(
                    children=child1,
                    features_groups=feature_groups,
                    mutation_prob=self.mprob,
                )
                child2 = self.mutation(
                    children=child2,
                    features_groups=feature_groups,
                    mutation_prob=self.mprob,
                )

                children[2 * j] = child1
                children[2 * j + 1] = child2

            if self.elitism:
                childId = np.random.choice(self.popsize, 1)
                children[childId] = population[bestFitness_Id]

            population = children
            print("")

        end_time_GA = time.time()

        # Calculate the execution time in minutes
        execution_time_minutes = (end_time_GA - start_time_GA) / 60

        sv = SaveResults(
            output=output,
            output_score=output_score,
            best_individual=bestInd,
            time_exec_ga=execution_time_minutes,
        )

        sv.save()
