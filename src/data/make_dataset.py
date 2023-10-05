# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

DATA_PATH = os.environ["DATA_PATH"]


def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """


if __name__ == "__main__":
    df = pd.read_csv(f"{DATA_PATH}/interim/dataset.csv")

    print(df)
