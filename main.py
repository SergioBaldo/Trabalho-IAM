from src.genetic_algorithm.genetic_search import GeneticSearch
from src.wrapper.wrapper import Wrapper
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Load environment variables
load_dotenv(find_dotenv(), override=True)
DATA_PATH = os.environ.get("DATA_PATH")
SEED = int(os.environ.get("SEED"))


df = pd.read_csv(f"{DATA_PATH}/processed/data_train.csv", index_col=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)

ga = GeneticSearch()

wrapper = Wrapper(clf=clf, sm=ga, X=X, y=y)

wrapper.select(X=X, y=y)
