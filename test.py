from src.genetic_algorithm.utils import create_feature_groups, run_best_individual
import numpy as np
from tqdm import trange
from dotenv import load_dotenv, find_dotenv
from src.genetic_algorithm import ga_functions as ga
from src.genetic_algorithm.save_ga_results import SaveResults
import os 
import pandas as pd 

load_dotenv(find_dotenv(), override=True)
DATA_PATH = os.environ.get("DATA_PATH")
RESULTS_PATH = os.environ.get("RESULTS_PATH")
SEED = int(os.environ.get("SEED"))


np.random.seed(SEED)
df_train = pd.read_csv(DATA_PATH + "/processed/data_train.csv")

feature_names = list(df_train.columns)
feature_names.remove("subtype")


run_best_individual(data_path=DATA_PATH, best_individual=feature_names, generation=None)
