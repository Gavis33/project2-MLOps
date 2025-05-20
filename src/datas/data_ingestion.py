import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import yaml
import logging
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from src.logger import logging
from src.connections import s3_connection
from sklearn.model_selection import train_test_split


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise 

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logging.info(f"Data loaded from {data_url} successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].map({'positive':1, 'negative':0}).infer_objects()
        return final_df
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logging.info(f"Data saved successfully at {raw_data_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2

        df = load_data(data_url='./data/IMDB.csv')
        # s3 = s3_connection.s3Operation('bucket-name', 'accesskey', 'secretkey')
        # df = s3.fetch_data_from_s3('data.csv')

        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()