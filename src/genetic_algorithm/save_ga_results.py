import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv, find_dotenv
import os
import json

load_dotenv(find_dotenv(), override=True)
POPSIZE = int(os.environ.get("POPSIZE"))
N_GENERATION = int(os.environ.get("N_GENERATION"))
CROSSOVER_PROB = float(os.environ.get("CROSSOVER_PROB"))
MUTATION_PROB = float(os.environ.get("MUTATION_PROB"))
TOUR_SIZE = int(os.environ.get("TOUR_SIZE"))
INDIVIDUAL_SIZE = int(os.environ.get("INDIVIDUAL_SIZE"))
ELITISM = eval(os.environ.get("ELITISM"))
SEED = int(os.environ.get("SEED"))
RESULTS_PATH = os.environ.get("RESULTS_PATH")


class SaveResults:
    def __init__(self, output, output_score, best_individual, time_exec_ga):
        """
        Initialize the SaveResults object.

        Args:
            output (numpy.ndarray): The fitness values for each generation.
            output_score (numpy.ndarray): The best accuracy scores for each generation.
            best_individual (numpy.ndarray): The best individual.
            time_exec_ga (float): The time it took to execute the genetic algorithm
        """
        self.output = output
        self.output_score = output_score
        self.best_individual = best_individual
        self.time_exec_ga = time_exec_ga
        self.path = RESULTS_PATH + f"/seed_{SEED}_IndividualSize_{INDIVIDUAL_SIZE}"

    def plot_mean_fitness_generation(self):
        """
        Plot the mean fitness (Weighted Averaged F1 Score) for each generation and save it as a PDF.
        """
        plt.figure()
        plt.plot(np.arange(N_GENERATION), self.output[:, 0], "ko-")
        plt.title("Mean Fitness (Weighted Averaged F1 Score) in Each Generation")
        plt.xlabel("Generations")
        plt.ylabel("Mean")
        plt.tight_layout()
        plt.savefig(
            self.path + "/MeanFitnessInEachGeneration.pdf",
            format="pdf",
        )
        plt.clf()

    def plot_std_fitness_generation(self):
        """
        Plot the standard deviation of fitness (Weighted Averaged F1 Score) for each generation and save it as a PDF.
        """
        plt.figure()
        plt.plot(np.arange(N_GENERATION), self.output[:, 1], "ko-")
        plt.title("Standard Deviation of Fitness (Weighted Averaged F1 Score) in Each Generation")
        plt.xlabel("Generations")
        plt.ylabel("Standard Deviation")
        plt.tight_layout()
        plt.savefig(
            self.path + "/StdDevFitnessInEachGeneration.pdf",
            format="pdf",
        )
        plt.clf()

    def best_fitness_generation(self):
        """
        Plot the best fitness (Weighted Averaged F1 Score) for each generation and save it as a PDF.
        """
        plt.figure()
        plt.plot(np.arange(N_GENERATION), self.output[:, 2], "ko-")
        plt.title("Best Fitness (Weighted Averaged F1 Score) in Each Generation")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.tight_layout()
        plt.savefig(
            self.path + "/BestFitnessInEachGeneration.pdf",
            format="pdf",
        )
        plt.clf()

    def best_accuracy_generation(self):
        """
        Plot the best accuracy for each generation and save it as a PDF.
        """
        plt.figure()
        plt.plot(np.arange(N_GENERATION), self.output_score, "ko-")
        plt.title("Best Accuracy in Each Generation")
        plt.xlabel("Generations")
        plt.ylabel("Balanced Accuracy")
        plt.tight_layout()
        plt.savefig(
            self.path + "/BestAccuracyInEachGeneration.pdf",
            format="pdf",
        )
        plt.clf()

    def save_txt_files(self):
        """
        Save fitness information to a text file.
        """
        np.savetxt(
            self.path + "/FitnessGenInfo.txt",
            self.output,
            delimiter=",",
            fmt="%.2f",
            header="MeanFitness, StdFitness, MaxFitness",
            comments="",
        )

    def save_features_of_the_best_individual(self):
        """
        Save features of the best individual to a text file.
        """
        np.savetxt(
            self.path + "/BestIndividual.txt",
            self.best_individual,
            delimiter=",",
            fmt="%s",
        )

    def save_ga_info(self):
        """
        Save genetic algorithm information to a JSON-formatted text file.
        """
        info = {
            "Popsize": POPSIZE,
            "Ngenerations": N_GENERATION,
            "Crossover_prob": CROSSOVER_PROB,
            "Mutation_prob": MUTATION_PROB,
            "Tour_size": TOUR_SIZE,
            "Individual_size": INDIVIDUAL_SIZE,
            "Elitism": ELITISM,
            "seed": SEED,
            "Time_to_exec_GA": str(f"{self.time_exec_ga:.2f} min"),
        }

        # Convert the dictionary to a JSON string
        json_str = json.dumps(
            info, indent=4
        )  # 'indent' is optional and makes the output more readable

        # Open the file in write mode and write the JSON string to it
        with open(self.path + "/GA_info.txt", "w") as file:
            file.write(json_str)

    def save(self):
        """
        Save function to save results and generate plots.
        """

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.plot_mean_fitness_generation()
        self.plot_std_fitness_generation()
        self.best_fitness_generation()
        self.best_accuracy_generation()
        self.save_txt_files()
        self.save_features_of_the_best_individual()
        self.save_ga_info()
