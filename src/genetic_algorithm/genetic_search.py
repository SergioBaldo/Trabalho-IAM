import os
import time
import pandas as pd
import numpy as np
import random
from tqdm import trange
from dotenv import load_dotenv, find_dotenv
from src.genetic_algorithm import ga_functions as ga
from src.genetic_algorithm.utils import create_feature_groups, run_best_individual
from src.genetic_algorithm.save_ga_results import SaveResults

load_dotenv(find_dotenv(), override=True)
DATA_PATH = os.environ.get("DATA_PATH")
RESULTS_PATH = os.environ.get("RESULTS_PATH")
SEED = int(os.environ.get("SEED"))


np.random.seed(SEED)
random.seed(SEED)


class GeneticSearch:
    def __init__(
        self,
        popsize: int,
        ngeneration: int,
        cprob: float,
        mprob: float,
        tour_size: int,
        individual_size: int,
        elitism: bool,
    ):
        """
        Initialize the GeneticSearch class with parameters.

        Args:
            popsize (int): Population size.
            ngeneration (int): Number of generations.
            cprob (float): Crossover probability.
            mprob (float): Mutation rate.
            tour_size (int): Tournament size for selection.
            individual_size (int): Size of each individual.
            elitism (bool): Whether to use elitism.
        """
        self.popsize = popsize
        self.ngeneration = ngeneration
        self.cprob = cprob
        self.mprob = mprob
        self.tour_size = tour_size
        self.individual_size = individual_size
        self.elitism = elitism

    # def open_data(self, filepath: str) -> tuple:
    #     """
    #     Load and return training and validation data.

    #     Args:
    #         filepath (str): Path to data files.

    #     Returns:
    #         tuple: A tuple containing two dataframes representing training and validation data.
    #     """
    #     filepath = f"{filepath}/processed/"
    #     df_train_GA = pd.read_csv(f"{filepath}dataGA_train.csv", index_col=None)
    #     df_validation_GA = pd.read_csv(f"{filepath}dataGA_test.csv", index_col=None)
    #     return df_train_GA, df_validation_GA
    
    def open_data(self, filepath: str) -> tuple:
        """
        Load and return training and validation data.

        Args:
            filepath (str): Path to data files.

        Returns:
            tuple: A tuple containing two dataframes representing training and validation data.
        """
        filepath = f"{filepath}/processed/"
        df = pd.read_csv(f"{filepath}data_train.csv", index_col=None)

        return df 

    def search(self):
        """
        Perform genetic search.

        This method initializes populations, calculates fitness, and evolves populations over generations.
        """
        df_train_GA = self.open_data(filepath=DATA_PATH)
        feature_names = list(df_train_GA.columns)
        feature_groups = create_feature_groups(
            feature_names=feature_names, individual_size=self.individual_size
        )

        start_time_AG = time.time()

        # Generate a population with M individuals (initial population)
        population = np.zeros((self.popsize, self.individual_size), dtype="O")

        fitness_has_table = {}
        output = np.zeros((self.ngeneration, 3))
        output_score = np.zeros(self.ngeneration)
        bestFitness = float("-inf")

        # Generate individuals
        for i in range(self.popsize):
            population[i] = ga.create_individual(
                individual_size=self.individual_size, feature_names=feature_names
            )

        if not os.path.isdir(RESULTS_PATH + f"/SEED_{SEED}"):
            os.mkdir(RESULTS_PATH + f"/SEED_{SEED}")

        for gen in range(self.ngeneration):
            print("")
            fitness = np.zeros(self.popsize)
            accuracy = np.zeros(self.popsize)

            # for individual in trange(self.popsize, desc=f"Generation: {gen}"):
            #     accuracy[individual], fitness[individual] = ga.fitness_function(
            #         individual=population[individual],
            #         df_train=df_train_GA,
            #         df_validation=df_validation_GA,
            #         fitness_hash_table=fitness_has_table,
            #     )
            for individual in trange(self.popsize, desc=f"Generation: {gen}"):
                accuracy[individual], fitness[individual] = ga.fitness_function(
                    individual=population[individual],
                    df=df_train_GA,
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
            run_best_individual(data_path=DATA_PATH, best_individual=bestInd, generation=gen)

            children = np.zeros((self.popsize, self.individual_size), dtype="O")
            for j in range(int(self.popsize / 2)):
                parent1, parent2 = ga.tournament_selection(
                    fitness=fitness, population=population, k_tour=self.tour_size
                )

                child1, child2 = ga.two_point_crossover(
                    parent1=parent1, parent2=parent2, cprob=self.cprob
                )

                child1 = ga.nominal_mutation(
                    children=child1,
                    features_groups=feature_groups,
                    mutation_prob=self.mprob,
                )
                child2 = ga.nominal_mutation(
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

        sv = SaveResults(
            results_path=RESULTS_PATH,
            ngenerations=self.ngeneration,
            output=output,
            output_score=output_score,
            best_individual=bestInd,
            seed=SEED,
        )

        sv.main()
