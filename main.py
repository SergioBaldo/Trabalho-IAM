# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
from src.genetic_algorithm.genetic_search import GeneticSearch
from src.wrapper.wrapper import Wrapper
from src.utils import feature_selection_and_evaluation

# Load environment variables
load_dotenv(find_dotenv(), override=True)
DATA_PATH = os.environ.get("DATA_PATH")
SEED = int(os.environ.get("SEED"))
RESULTS_PATH = os.environ.get("RESULTS_PATH")
INDIVIDUAL_SIZE = int(os.environ.get("INDIVIDUAL_SIZE"))


if __name__ == "__main__":
    # Load and preprocess the training data
    df = pd.read_csv(f"{DATA_PATH}/processed/data_train.csv", index_col=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Initialize the Genetic Algorithm search
    ga = GeneticSearch()

    # Initialize the Wrapper
    wrapper = Wrapper(clf=clf, sm=ga, X=X, y=y)

    # Feature selection using the Wrapper
    wrapper.select(X=X, y=y)

    # Run different models with various feature selection methods
    feature_selection_and_evaluation.run_models(clf)
