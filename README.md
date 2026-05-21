# Customer Churn Prediction System

## End-to-End Production Grade Machine Learning Project with MLOps & Cloud Deployment

---

# Overview

This project is a production-grade end-to-end Machine Learning system for predicting customer churn using modern MLOps and cloud-native deployment practices.

The system integrates:

- FastAPI for model serving
- Docker for containerization
- DVC for data version control
- MLflow for experiment tracking
- AWS S3 for artifact storage
- AWS ECR for container registry
- AWS EC2 for deployment
- GitHub Actions for CI/CD automation

The project is designed to demonstrate enterprise-level Machine Learning Engineering and MLOps capabilities.

---

# Business Problem

Customer churn is one of the most important challenges for subscription-based businesses.

Companies lose significant revenue when customers discontinue their services.

The goal of this system is to:

- Predict customer churn probability
- Enable proactive retention strategies
- Reduce customer attrition
- Improve customer lifetime value
- Optimize retention campaigns

Industries where churn prediction is widely used:

- Telecommunications
- Banking
- Insurance
- SaaS Platforms
- E-commerce
- Subscription Services

---

# Key Features

## Machine Learning

- End-to-end ML pipeline
- Feature engineering
- Data preprocessing
- Class imbalance handling using SMOTE
- Model training and evaluation
- Automated inference pipeline

## MLOps

- Data versioning with DVC
- Experiment tracking with MLflow
- Artifact versioning using AWS S3
- Modular pipeline architecture
- Reproducible workflows

## Deployment

- Dockerized FastAPI application
- AWS cloud deployment
- GitHub Actions CI/CD pipeline
- Automated redeployment
- Health monitoring

---

# System Architecture

```text
                    ┌────────────────────┐
                    │   GitHub Repository │
                    └─────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ GitHub Actions CI/CD │
                   └─────────┬───────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
 ┌─────────────────┐                 ┌─────────────────┐
 │ Docker Build     │                 │ Model Artifacts │
 │ Push to ECR      │                 │ Stored in S3    │
 └────────┬────────┘                 └────────┬────────┘
          │                                    │
          ▼                                    ▼
 ┌────────────────────────────────────────────────────┐
 │                  EC2 Deployment                     │
 │                                                    │
 │  FastAPI + Docker Container                        │
 │  Downloads Artifacts from S3                       │
 │  Loads Trained Model                               │
 └────────────────────────────────────────────────────┘
```

---

# Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.11 |
| API Framework | FastAPI |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Containerization | Docker |
| Cloud Storage | AWS S3 |
| Container Registry | AWS ECR |
| Deployment Server | AWS EC2 |
| CI/CD | GitHub Actions |
| Cloud Platform | AWS |

---

# Machine Learning Pipeline

The project uses a modular Machine Learning pipeline.

## 1. Data Ingestion

- Load raw dataset
- Store ingestion artifacts
- Validate file availability

## 2. Data Validation

- Schema validation
- Missing value checks
- Data consistency checks
- Drift monitoring

## 3. Data Preprocessing and Transformation

- Feature engineering
- Categorical encoding
- Numerical scaling
- SMOTE oversampling
- Pipeline creation

## 4. Model Training

Models trained:

- Logistic Regression
- XGBoost Classifier

## 5. Model Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# Project Structure

```text
END_TO_END_ML_PROJECT_FOR_CUSTOMER_CHURN_PREDICTION/
│
├── .github/
│   └── workflows/
│       └── aws.yaml
│
├── artifacts/
│
├── config/
│   └── config.yaml
│
├── research/
│
├── src/
│   └── customer_churn_prediction/
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── pipeline/
│       ├── utils/
│
│
├── templates/
│
├── app.py
├── main.py
├── Dockerfile
├── requirements.txt
├── setup.py
├── dvc.yaml
└── params.yaml
```

---

# Dataset Description

The dataset contains customer-related information such as:

- Gender
- Senior Citizen Status
- Partner Status
- Dependents
- Contract Type
- Internet Service
- Payment Method
- Monthly Charges
- Total Charges
- Tenure

## Target Variable

```text
Churn Label / Churn Value
```

- Yes →  1 -> Customer leaves
- No → 0 -> Customer stays

---

# Data Versioning with DVC

DVC is used to version datasets and ML artifacts.

## Initialize DVC

```bash
dvc init
```

## Add Dataset

```bash
dvc add data/customer_churn.csv
```

## Configure S3 Remote Storage

```bash
dvc remote add -d s3remote s3://customer-churn-models
```

## Push Data to Remote

```bash
dvc push
```

---

# Experiment Tracking with MLflow

MLflow is used for:

- Experiment tracking
- Parameter logging
- Metric comparison
- Model versioning

## Start MLflow UI

```bash
mlflow ui
```

Access:

```text
http://localhost:5000
```

---

# Model Training

Run complete training pipeline:

```bash
python main.py
```

Generated artifacts:

```text
artifacts/
├── data_transformation/
│   └── preprocessor.pkl
└── model_training/
    └── xgboost_model.pkl
```

---

# AWS S3 Artifact Storage

Model artifacts are stored in AWS S3.

## Why S3?

- Centralized storage
- Scalable artifact management
- Production-grade architecture
- Decouples model from Docker image

## S3 Folder Structure

```text
s3://customer-churn-models/
└── artifacts/
    ├── data_transformation/
    │   └── preprocessor.pkl
    └── model_training/
        └── xgboost_model.pkl
```

---

# FastAPI Inference Service

FastAPI exposes REST APIs for model inference.

## Run API Locally

```bash
uvicorn app:app --reload
```

## API Documentation

```text
http://localhost:8000/docs
```

---

# API Endpoints

## Health Check

```http
GET /health
```

### Response

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## Predict Customer Churn

```http
POST /predict
```

### Sample Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "tenure": 12,
  "MonthlyCharges": 70.5
}
```

### Sample Response

```json
{
  "prediction": "Yes",
  "probability": 0.87
}
```

---

# Docker Containerization

Docker is used for packaging the application.

## Build Docker Image

```bash
docker build -t churn-prediction:latest .
```

## Run Docker Container

```bash
docker run -d -p 8000:8000 churn-prediction:latest
```

---

# AWS Deployment Architecture

The application is deployed using:

| AWS Service | Purpose |
|---|---|
| EC2 | Application Hosting |
| ECR | Docker Image Registry |
| S3 | Artifact Storage |
| IAM | Security & Permissions |
| GitHub Actions | CI/CD Automation |

---

# Continuous Integration & Continuous Deployment

The CI/CD pipeline automatically:

1. Runs integration checks
2. Builds Docker image
3. Pushes image to ECR
4. Pulls latest image on EC2
5. Stops previous container
6. Deploys updated container
7. Runs health checks

---

# GitHub Actions Workflow

## CI/CD Pipeline Stages

```text
Code Push
    ↓
Continuous Integration
    ↓
Docker Build
    ↓
Push Image to ECR
    ↓
Deploy on EC2
    ↓
Health Check
```

---

# Installation Guide

## Clone Repository

```bash
git clone git@github.com:ajaychaudhary8104/End_to_End_ML_project_for_Customer_Churn_Prediction.git
```

## Create Virtual Environment

```bash
conda create -n customer python=3.11
conda activate customer
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Local Development

## Run Training Pipeline

```bash
python main.py
```

## Run FastAPI Application

```bash
uvicorn app:app --reload
```

---

# Docker Deployment

## Build Docker Image

```bash
docker build -t churn-prediction:latest .
```

## Run Docker Container

```bash
docker run -d -p 8000:8000 churn-prediction:latest
```

---

# AWS EC2 Deployment

## Pull Docker Image

```bash
docker pull <your-ecr-image-uri>
```

## Run Docker Container

```bash
docker run -d -p 8000:8000 <your-ecr-image-uri>
```

---

# Monitoring & Logging

Application logs include:

- API startup logs
- Artifact download status
- Model loading status
- Prediction logs
- Exception traces
- Health check status

---

# Security Best Practices

The project follows cloud security best practices:

- IAM least privilege policies
- Environment-based secret management
- GitHub Secrets integration
- Secure S3 access
- Docker isolation
- Non-hardcoded credentials

---

# Future Improvements

Potential future enhancements:

- Kubernetes deployment
- Auto-scaling
- Real-time monitoring dashboards
- Model drift detection
- Automated retraining
- Blue-green deployments
- Canary deployments
- Model registry integration
- Feature store integration

---

# Business Impact

This solution helps organizations:

- Reduce churn rates
- Increase revenue retention
- Improve customer engagement
- Enable proactive intervention
- Improve customer satisfaction
- Optimize marketing strategies

---

# Configuration Workflow

To modify the pipeline:

1. Update `config/config.yaml` - Set paths and parameters
2. Update `params.yaml` - Modify hyperparameters
3. Update `entity/config_entity.py` - Define configuration entities
4. Update `config/configuration.py` - Implement configuration manager
5. Update components in `components/` - Modify pipeline stages
6. Update pipeline stages in `pipeline/` - Update pipeline logic
7. Update `main.py` - Execute the pipeline
8. Update `dvc.yaml` - Define DVC pipeline stages


set MLFLOW_TRACKING_URI=https://dagshub.com/ajaychaudhary8104/End_to_End_ML_project_for_Customer_Churn_Prediction.mlflow
set MLFLOW_TRACKING_USERNAME=ajaychaudhary8104
set MLFLOW_TRACKING_PASSWORD=

docker build -t churn-prediction:latest .
docker run -p 8000:8000 churn-prediction:latest


# AWS CI/CD Deployment with GitHub Actions

## Step 1: Login to AWS Console

Go to:

[AWS Console](https://aws.amazon.com/console/?utm_source=chatgpt.com)

---

# Step 2: Create IAM User for Deployment

Go to:

* IAM → Users → Create User

## Required Permissions

Attach these policies:

* `AmazonEC2FullAccess`
* `AmazonEC2ContainerRegistryFullAccess`

## Purpose of These Permissions

### EC2 Access

Used to manage virtual machines.

### ECR Access

Used to store Docker images in AWS Elastic Container Registry.

---

# CI/CD Deployment Flow

1. Build Docker image from source code
2. Push Docker image to Amazon ECR
3. Launch EC2 instance
4. Pull Docker image from ECR inside EC2
5. Run Docker container on EC2

---

# Step 3: Create ECR Repository

Go to:

* Elastic Container Registry (ECR)
* Create Repository

Example Repository URI:

```bash
577124149610.dkr.ecr.us-east-1.amazonaws.com/customer
```

Save this URI for GitHub Secrets.

---

# Step 4: Launch EC2 Instance

Recommended Configuration:

* Ubuntu Server 22.04
* t2.medium or higher
* Minimum 20GB storage

Allow These Inbound Rules:

| Type       | Port |
| ---------- | ---- |
| SSH        | 22   |
| HTTP       | 80   |
| HTTPS      | 443  |
| Custom TCP | 8000 |

---

# Step 5: Install Docker on EC2

Connect to EC2:

```bash
ssh -i key.pem ubuntu@<EC2_PUBLIC_IP>
```

Run:

```bash
sudo apt-get update -y

sudo apt-get upgrade -y
```

Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh
```

Add Ubuntu user to Docker group:

```bash
sudo usermod -aG docker ubuntu
```

Activate group changes:

```bash
newgrp docker
```

Verify Docker:

```bash
docker --version
```

---

# Step 6: Configure EC2 as GitHub Self-Hosted Runner

Go to your GitHub repository:

```text
Settings → Actions → Runners → New Self-hosted Runner
```

Choose:

* Linux
* x64

Run all commands provided by GitHub one-by-one on EC2.

Example:

```bash
mkdir actions-runner && cd actions-runner

curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-x64-2.317.0.tar.gz

tar xzf ./actions-runner-linux-x64.tar.gz
```

Configure runner:

```bash
./config.sh --url https://github.com/<username>/<repo> --token <TOKEN>
```

Start runner:

```bash
./run.sh
```

For background service:

```bash
sudo ./svc.sh install

sudo ./svc.sh start
```

---

# Step 7: Configure GitHub Secrets

Go to:

```text
Repository → Settings → Secrets and variables → Actions
```

Add:

```bash
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_DEFAULT_REGION=us-east-1

AWS_ECR_LOGIN_URI=

ECR_REPOSITORY_NAME=customer
```

---

# Recommended Project Structure

```text
project/
│
├── .github/
│   └── workflows/
│       └── main.yaml
│
├── Dockerfile
├── requirements.txt
├── app.py
├── src/
├── templates/
├── static/
└── README.md
```

---

# Verify Deployment

After GitHub Actions succeeds:

Check running containers:

```bash
docker ps
```

Open:

```text
http://<EC2_PUBLIC_IP>:8080
```

---

# Useful Docker Commands

Stop container:

```bash
docker stop cnncls
```

Remove container:

```bash
docker rm cnncls
```

Remove images:

```bash
docker image prune -a
```

View logs:

```bash
docker logs -f cnncls
```

---

# Conclusion

This project demonstrates a complete production-grade Machine Learning system integrating:

- Machine Learning Engineering
- MLOps Practices
- Cloud Deployment
- CI/CD Automation
- Scalable API Deployment

The architecture is modular, scalable, maintainable, and enterprise-ready for real-world production deployment.