import boto3
import os

S3_BUCKET = "customer-churn-models"

FILES_TO_UPLOAD = [
    "artifacts/data_transformation/preprocessor.pkl",
    "artifacts/model_training/xgboost_model.pkl"
]

s3 = boto3.client("s3")

for file_path in FILES_TO_UPLOAD:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    s3.upload_file(
        Filename=file_path,
        Bucket=S3_BUCKET,
        Key=file_path
    )

    print(f"Uploaded: {file_path}")