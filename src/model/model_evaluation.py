import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import yaml
import json
import mlflow
import pickle
import logging
import dagshub
import numpy as np
import pandas as pd
from src.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Production code
# dagshub_token = os.getenv("PROJECT2_TEST")
# if not dagshub_token:
#     raise ValueError("Dagshub token not found. Please set the 'PROJECT2_TEST' environment variable.")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com/Gavis33/project2-MLOps.mlflow"
# mlflow.set_tracking_uri(dagshub_url)

# local code
dagshub_url = "https://dagshub.com/Gavis33/project2-MLOps.mlflow"
mlflow.set_tracking_uri(dagshub_url)
dagshub.init(repo_owner="Gavis33", repo_name="project2-MLOps", mlflow=True)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f'Model loaded from {file_path}')
        return model
    except Exception as e:
        logging.error(f'Unexpected error occurred while loading the model: {e}')
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded and NaNs filled from {file_path}')
        return df
    except Exception as e:
        logging.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

        logging.info('Model evaluated successfully')
        return metrics
    except Exception as e:
        logging.error(f'Unexpected error occurred while evaluating the model: {e}')
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info(f'Metrics saved to {file_path}')
    except Exception as e:
        logging.error(f'Unexpected error occurred while saving the metrics: {e}')
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info(f'Model info saved to {file_path}')
    except Exception as e:
        logging.error(f'Unexpected error occurred while saving the model info: {e}')
        raise

def main():
    mlflow.set_experiment("project2-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            model = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(model, X_test, y_test)

            save_metrics(metrics, 'reports/metrics.json')

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                
            mlflow.sklearn.log_model(model, "model")

            save_model_info(run.info.run_id, 'model', 'reports/experiment_info.json')

            mlflow.log_artifact('reports/metrics.json')
            
        except Exception as e:
            logging.error(f'Failed to evaluate the model: {e}')
            raise

if __name__  == "__main__":
    main()