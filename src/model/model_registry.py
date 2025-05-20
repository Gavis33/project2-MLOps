import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import json
import mlflow
import logging
import dagshub
from src.logger import logging

import warnings 
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Production code
# dagshub_token = os.getenv("PROJECT2_TEST")
# if not dagshub_token:
#     raise EnvironmentError('Dagshub token not found. Please set the "PROJECT2_TEST" environment variable.')

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com/Gavis33/project2-MLOps.mlflow"
# mlflow.set_tracking_uri(dagshub_url)

# local code
dagshub_url = "https://dagshub.com/Gavis33/project2-MLOps.mlflow"
mlflow.set_tracking_uri(dagshub_url)
dagshub.init(repo_owner="Gavis33", repo_name="project2-MLOps", mlflow=True)

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info(f'Model info loaded from {file_path}')
        return model_info
    except Exception as e:
        logging.error(f'Unexpected error occurred while loading the model info: {e}')
        raise

def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.info(f'Model registered with name {model_name}')
    
    except Exception as e:
        logging.error(f'Unexpected error occurred while registering the model: {e}')
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = 'project2_model'
        register_model(model_name, model_info)

    except Exception as e:
        logging.error(f'Failed to register the model: {e}')
        raise

if __name__ == "__main__":
    main()