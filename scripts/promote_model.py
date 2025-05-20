import os
import mlflow

def promote_model():
    dagshub_token = os.getenv("PROJECT2_TEST")
    if not dagshub_token:
        raise EnvironmentError('Dagshub token not found. Please set the "PROJECT2_TEST" environment variable.')

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri('https://dagshub.com/Gavis33/project2-MLOps.mlflow')

    client = mlflow.MlflowClient()
    model_name = 'project2_model'

    latest_version_in_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    for production_version in production_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=production_version.version,
            stage="Archived"
        )

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_in_staging,
        stage="Production"
        )

    print(f'Model version {latest_version_in_staging} promoted to Production')

if __name__ == "__main__":
    promote_model()