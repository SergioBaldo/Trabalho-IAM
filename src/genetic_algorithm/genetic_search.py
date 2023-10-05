import numpy as np
import time
import os
from dotenv import load_dotenv
from ga_functions import create_individual, fitness_function

# Load environment variables
load_dotenv()
POPSIZE = int(os.environ.get("POPSIZE"))
N_GENERATION = int(os.environ.get("N_GENERATION"))
CROSSOVER_PROB = float(os.environ.get("CROSSOVER_PROB"))
TOUR_SIZE = int(os.environ.get("TOUR_SIZE"))
SELECTION = os.environ.get("SELECTION")
INDIVIDUAL_SIZE = int(os.environ.get("INDIVIDUAL_SIZE"))


class GeneticSearch:
    def __init__(
        self,
        popsize=POPSIZE,
        ngeneration=N_GENERATION,
        cprob=CROSSOVER_PROB,
        tour_size=TOUR_SIZE,
        selection=SELECTION,
        individual_size=INDIVIDUAL_SIZE,
    ):
        """
        Initialize the GeneticSearch class with parameters.

        Args:
            popsize (int): Population size.
            ngeneration (int): Number of generations.
            cprob (float): Crossover probability.
            tour_size (int): Tournament size for selection.
            selection (str): Selection method.
            individual_size (int): Size of each individual.
        """
        self.popsize = popsize
        self.ngeneration = ngeneration
        self.cprob = cprob
        self.tour_size = tour_size
        self.selection = selection
        self.individual_size = individual_size

    def open_data(self):
        """
        Load and return training and validation data.

        Returns:
            tuple: A tuple containing two empty lists representing training and validation data.
        """
        df_train_GA = []
        df_validation_GA = []
        return df_train_GA, df_validation_GA

    def search(self):
        """
        Perform genetic search.

        This method initializes populations, calculates fitness, and evolves populations over generations.
        """
        df_train_GA, df_validation_GA = self.open_data()

        feature_names = [f"feature{i}" for i in range(20000)]

        start_time_AG = time.time()

        fitness_has_table = {}

        # Generate a population with M individuals (initial population)
        population = np.zeros((self.popsize, self.individual_size), dtype="O")

        output = np.zeros((self.ngeneration, 4))
        output_score = np.zeros(self.ngeneration)
        bestFitness = float("-inf")

        # Generate individuals
        for i in range(self.popsize):
            population[i] = create_individual(
                individual_size=self.individual_size, feature_names=feature_names
            )

        for gen in range(self.ngeneration):
            print(f"Generation: {gen}")

            fitness = np.zeros(self.popsize)
            score = np.zeros(self.popsize)

            # Calculate the fitness of each individual
            for individual in range(self.popsize):
                score[individual], fitness[individual] = fitness_function(
                    individual=population[individual],
                    df_train=df_train_GA,
                    df_validation=df_validation_GA,
                    fitness_has_table=fitness_has_table,
                )

            output_score[gen] = np.max(score)
            output[gen, 0] = np.mean(fitness)  # Mean fitness of the population
            output[gen, 1] = np.std(fitness)  # Fitness standard deviation
            output[gen, 2] = np.max(fitness)  # Best fitness

            bestFitness_Id = np.argmax(fitness)

            if bestFitness < np.max(fitness):
                bestInd = population[bestFitness_Id]
                bestFitness = np.max(fitness)

            print(f"Best Fitness gen. {gen}: {bestFitness}")

            # Generate M children
            children = np.zeros((self.popsize, self.individual_size), dtype="O")
            for j in range(int(self.popsize / 2)):
                # Selection
                parent1, parent2 = self.selection(self.tour_size, fitness, population)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2, self.cprob)

                # Mutation
                child1 = self.mutation(child1, prob=self.cprob, nfeature=self.nFeatures)
                child2 = self.mutation(child2, prob=self.cprob, nfeature=self.nFeatures)

                children[2 * j] = child1
                children[2 * j + 1] = child2

            # Elitism
            if self.elitism:
                childId = np.random.choice(self.popsize, 1)
                children[childId] = population[bestFitness_Id]

            population = children
