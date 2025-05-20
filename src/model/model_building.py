import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd

import pickle 
import yaml
from src.logger import logging
from sklearn.linear_model import LogisticRegression

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded and NaNs filled from {file_path}')
        return df
    except Exception as e:
        logging.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    try:
        model = LogisticRegression(C=1, solver='liblinear', penalty='l2') 
        # where C is the inverse of regularization strength (in other words, higher C means more regularization / protection from overfitting)
        # solver is the algorithm used to solve the optimization problem (liblinear is a good choice for small datasets)
        # penalty is the type of regularization (L1 or L2) (L2 is the default and is often used for classification problems)
        model.fit(X_train, y_train)
        logging.info('Model trained successfully')
        return model
    except Exception as e:
        logging.error(f'Unexpected error occurred while training the model: {e}')
        raise

def save_model(model: LogisticRegression, file_path: str) -> None:
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f'Model saved to {file_path}')
    except Exception as e:
        logging.error(f'Unexpected error occurred while saving the model: {e}')
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        model = train_model(X_train, y_train)
        save_model(model, './models/model.pkl')

        logging.info('Model built and saved successfully')
    except Exception as e:
        logging.error(f'Failed to build and save the model: {e}')
        raise

if __name__ == "__main__":
    main()