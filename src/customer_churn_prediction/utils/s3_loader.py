import boto3
import os
from src.customer_churn_prediction import logger

S3_BUCKET = os.getenv("S3_BUCKET_NAME")

FILES = [
    "artifacts/data_transformation/preprocessor.pkl",
    "artifacts/model_training/xgboost_model.pkl"
]

def download_artifacts():

    s3 = boto3.client("s3")

    for file_key in FILES:

        local_path = file_key

        if os.path.exists(local_path):

            logger.info(
                f"Artifact already exists: {local_path}"
            )

            continue

        os.makedirs(
            os.path.dirname(local_path),
            exist_ok=True
        )

        logger.info(f"Downloading {file_key}")

        s3.download_file(
            S3_BUCKET,
            file_key,
            local_path
        )

        logger.info(f"Downloaded {file_key}")