import matplotlib.pyplot as plt
import numpy as np


class SaveResults:
    def __init__(self, results_path, ngenerations, output, output_score, best_individual, seed):
        """
        Initialize the SaveResults object.

        Args:
            results_path (str): The path to save the results.
            ngenerations (int): The number of generations.
            output (numpy.ndarray): The fitness values for each generation.
            output_score (numpy.ndarray): The best accuracy scores for each generation.
            best_individual (numpy.ndarray): The best individual.
            seed (int): The seed value for the random generator.
        """
        self.results_path = results_path
        self.ngenerations = ngenerations
        self.output = output
        self.output_score = output_score
        self.best_individual = best_individual
        self.seed = seed

    def plot_mean_fitness_generation(self):
        """
        Plot the mean fitness (Weighted Averaged F1 Score) for each generation.
        """
        plt.figure()
        plt.plot(np.arange(self.ngenerations), self.output[:, 0], "ko-")
        plt.title("Mean Fitness (Weighted Averaged F1 Score) in Each Generation")
        plt.xlabel("Generations")
        plt.ylabel("Mean")
        plt.tight_layout()
        plt.savefig(
            self.results_path + f"/SEED_{self.seed}/MeanFitnessInEachGeneration.pdf", format="pdf"
        )
        plt.clf()

    def plot_std_fitness_generation(self):
        """
        Plot the standard deviation of fitness (Weighted Averaged F1 Score) for each generation.
        """
        plt.figure()
        plt.plot(np.arange(self.ngenerations), self.output[:, 1], "ko-")
        plt.title("Standard Deviation of Fitness (Weighted Averaged F1 Score) in Each Generation")
        plt.xlabel("Generations")
        plt.ylabel("Standard Deviation")
        plt.tight_layout()
        plt.savefig(
            self.results_path + f"/SEED_{self.seed}/StdDevFitnessInEachGeneration.pdf",
            format="pdf",
        )
        plt.clf()

    def best_fitness_generation(self):
        """
        Plot the best fitness (Weighted Averaged F1 Score) for each generation.
        """
        plt.figure()
        plt.plot(np.arange(self.ngenerations), self.output[:, 2], "ko-")
        plt.title("Best Fitness (Weighted Averaged F1 Score) in Each Generation")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.tight_layout()
        plt.savefig(
            self.results_path + f"/SEED_{self.seed}/BestFitnessInEachGeneration.pdf", format="pdf"
        )
        plt.clf()

    def best_accuracy_generation(self):
        """
        Plot the best accuracy for each generation.
        """
        plt.figure()
        plt.plot(np.arange(self.ngenerations), self.output_score, "ko-")
        plt.title("Best Accuracy in Each Generation")
        plt.xlabel("Generations")
        plt.ylabel("Balanced Accuracy")
        plt.tight_layout()
        plt.savefig(
            self.results_path + f"/SEED_{self.seed}/BestAccuracyInEachGeneration.pdf", format="pdf"
        )
        plt.clf()

    def save_txt_files(self):
        """
        Save fitness information to a text file.
        """
        np.savetxt(
            self.results_path + f"/SEED_{self.seed}/FitnessGenInfo.txt",
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
            self.results_path + f"/SEED_{self.seed}/BestIndividual.txt",
            self.best_individual,
            delimiter=",",
            fmt="%s",
        )

    def main(self):
        """
        Main function to save results and generate plots.
        """
        self.plot_mean_fitness_generation()
        self.plot_std_fitness_generation()
        self.best_fitness_generation()
        self.best_accuracy_generation()
        self.save_txt_files()
        self.save_features_of_the_best_individual()
