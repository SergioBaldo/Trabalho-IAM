import numpy as np


def roulette(k_tour, fitness, population):
    id_parents = np.random.choice(population.shape[0], 2, replace=False, p=fitness / sum(fitness))

    parent1 = population[id_parents[0]]
    parent2 = population[id_parents[1]]

    return parent1, parent2


def tournament(k_tour, fitness, population):
    id_parent1 = np.random.choice(population.shape[0], k_tour, replace=True)
    id_parent2 = np.random.choice(population.shape[0], k_tour, replace=True)

    idx1 = np.argmax(fitness[id_parent1])
    idx2 = np.argmax(fitness[id_parent2])

    parent1 = population[id_parent1[idx1]]
    parent2 = population[id_parent2[idx2]]

    return parent1, parent2


def crossover(parent1, parent2, cprob):
    if np.random.rand() < cprob:
        # crossover 2 points
        points_crossover = np.random.choice(np.arange(1, len(parent1)), 2, replace=True)

        if points_crossover[1] < points_crossover[0]:
            points_crossover[0], points_crossover[1] = points_crossover[1], points_crossover[0]

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
        children1 = parent1.copy()
        children2 = parent2.copy()

    return children1, children2
