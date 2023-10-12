from src.genetic_algorithm.genetic_search import GeneticSearch
from src.wrapper.wrapper import Wrapper
from dotenv import load_dotenv, find_dotenv
import os


# Load environment variables
load_dotenv(find_dotenv(), override=True)
POPSIZE = int(os.environ.get("POPSIZE"))
N_GENERATION = int(os.environ.get("N_GENERATION"))
CROSSOVER_PROB = float(os.environ.get("CROSSOVER_PROB"))
MUTATION_PROB = float(os.environ.get("MUTATION_PROB"))
TOUR_SIZE = int(os.environ.get("TOUR_SIZE"))
INDIVIDUAL_SIZE = int(os.environ.get("INDIVIDUAL_SIZE"))
ELITISM = eval(os.environ.get("ELITISM"))


ga = GeneticSearch(
    popsize=POPSIZE,
    ngeneration=N_GENERATION,
    cprob=CROSSOVER_PROB,
    mprob=MUTATION_PROB,
    tour_size=TOUR_SIZE,
    individual_size=INDIVIDUAL_SIZE,
    elitism=ELITISM,
)

wrapper = Wrapper(ga)

wrapper.select()
