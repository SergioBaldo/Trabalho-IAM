# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Define the data path using the environment variable
DATA_PATH = os.environ["DATA_PATH"]


def preprocess_and_split_data(input_filepath, output_filepath_prefix) -> None:
    """
    Load a dataset, preprocess it by splitting it into training and testing sets,
    and save the resulting datasets as separate CSV files.

    Args:
        input_filepath (str): The path to the input CSV file containing the dataset.
        output_filepath_prefix (str): The prefix for the output CSV file names.

    Returns:
        None
    """
    # Load the dataset from the input file
    df = pd.read_csv(input_filepath, index_col=0)

    # Separate the target column from the feature columns
    columns = list(df.columns)
    y_column_name = columns.pop()
    X = df[columns]
    y = df[y_column_name]

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Create DataFrames for the training and testing sets
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # Save the training and testing sets as separate CSV files
    train_output_filepath = f"{output_filepath_prefix}_train.csv"
    test_output_filepath = f"{output_filepath_prefix}_test.csv"
    df_train.to_csv(train_output_filepath, index=False)
    df_test.to_csv(test_output_filepath, index=False)


if __name__ == "__main__":
    # Define the paths for interim and processed data
    interim_path = f"{DATA_PATH}/interim/"
    processed_path = f"{DATA_PATH}/processed/"

    # Preprocess and split the initial dataset
    preprocess_and_split_data(
        input_filepath=f"{interim_path}dataset.csv", output_filepath_prefix=f"{processed_path}data"
    )

    # Preprocess and split the processed training dataset
    preprocess_and_split_data(
        input_filepath=f"{processed_path}data_train.csv",
        output_filepath_prefix=f"{processed_path}dataGA",
    )
