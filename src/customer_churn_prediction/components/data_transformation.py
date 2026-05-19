import os
import pickle
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.model_selection import (
    train_test_split
)
from src.customer_churn_prediction.entity.config_entity import DataTransformationConfig
from src.customer_churn_prediction import logger


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):

        self.config = config

        self.preprocessor = None

    # ======================================================
    # Load Cleaned Data
    # ======================================================

    def load_processed_data(self):

        if not os.path.exists(self.config.input_file_path):

            raise FileNotFoundError(
                f"Processed file not found: "
                f"{self.config.input_file_path}"
            )

        logger.info(
            f"Loading processed data from: "
            f"{self.config.input_file_path}"
        )

        data = pd.read_csv(
            self.config.input_file_path
        )

        logger.info(
            f"Processed data shape: "
            f"{data.shape}"
        )

        return data

    # ======================================================
    # Split Data
    # ======================================================

    def split_data(self, data):

        target_col = (
            self.config.target_column
        )

        X = data.drop(
            columns=[target_col]
        )

        y = data[target_col]

        X_train_val, X_test, y_train_val, y_test = (
            train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
        )

        relative_val_size = (
            self.config.validation_size /
            (1 - self.config.test_size)
        )

        X_train, X_val, y_train, y_val = (
            train_test_split(
                X_train_val,
                y_train_val,
                test_size=relative_val_size,
                random_state=self.config.random_state,
                stratify=y_train_val
            )
        )

        logger.info(
            f"Train shape: {X_train.shape}"
        )

        logger.info(
            f"Validation shape: {X_val.shape}"
        )

        logger.info(
            f"Test shape: {X_test.shape}"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

    # ======================================================
    # Build Preprocessor
    # ======================================================

    def build_preprocessor(
        self,
        numeric_columns,
        categorical_columns
    ):

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float32
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    numeric_columns
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )
            ],
            remainder="drop"
        )

        return preprocessor

    # ======================================================
    # Transform Data
    # ======================================================

    def transform_data(
        self,
        X_train,
        X_val,
        X_test
    ):
        
        # Numeric Columns
        numerical_columns = [
            col for col in self.config.numerical_columns
            if col in X_train.columns
        ]

    

        # Categorical Columns
        categorical_columns = [
            col for col in self.config.categorical_columns
            if col in X_train.columns
        ]

      

        logger.info(
            f"Numeric columns: {numerical_columns}"
        )

        logger.info(
            f"Categorical columns: {categorical_columns}"
        )
            

        self.preprocessor = (
            self.build_preprocessor(
                numerical_columns,
                categorical_columns
            )
        )

        logger.info(
            "Fitting preprocessor on train data"
        )

        X_train = (
            self.preprocessor.fit_transform(
                X_train
            )
        )

        X_val = (
            self.preprocessor.transform(
                X_val
            )
        )

        X_test = (
            self.preprocessor.transform(
                X_test
            )
        )

        feature_names = (
            self.preprocessor.get_feature_names_out()
        )

        X_train = pd.DataFrame(
            X_train,
            columns=feature_names
        )

        X_val = pd.DataFrame(
            X_val,
            columns=feature_names
        )

        X_test = pd.DataFrame(
            X_test,
            columns=feature_names
        )

        return X_train, X_val, X_test

    # ======================================================
    # Apply SMOTE
    # ======================================================

    def apply_smote(
        self,
        X_train,
        y_train
    ):

        logger.info(
            "Applying SMOTE..."
        )

        smote = SMOTE(
            random_state=self.config.random_state
        )

        X_resampled, y_resampled = (
            smote.fit_resample(
                X_train,
                y_train
            )
        )

        logger.info(
            f"After SMOTE: "
            f"{pd.Series(y_resampled).value_counts()}"
        )

        return X_resampled, y_resampled

    # ======================================================
    # Save Preprocessor
    # ======================================================

    def save_preprocessor(self):

        os.makedirs(
            self.config.root_dir,
            exist_ok=True
        )

        preprocessor_path = os.path.join(
            self.config.root_dir,
            "preprocessor.pkl"
        )

        with open(preprocessor_path, "wb") as file:

            pickle.dump(self.preprocessor,file)

        logger.info(
            f"Preprocessor saved to: "
            f"{preprocessor_path}"
        )

    # ======================================================
    # Save Data
    # ======================================================

    def save_data(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ):

        os.makedirs(
            self.config.split_artifacts_dir,
            exist_ok=True
        )

        train_df = X_train.copy()
        train_df[self.config.target_column] = y_train

        val_df = X_val.copy()
        val_df[self.config.target_column] = y_val.values

        test_df = X_test.copy()
        test_df[self.config.target_column] = y_test.values

        train_df.to_csv(
            self.config.train_file_path,
            index=False
        )

        val_df.to_csv(
            self.config.validation_file_path,
            index=False
        )

        test_df.to_csv(
            self.config.test_file_path,
            index=False
        )

        logger.info(
            "Train/Validation/Test saved"
        )

    # ======================================================
    # Main Pipeline
    # ======================================================

    def initiate_data_transformation(self):

        try:

            logger.info(
                "Starting data transformation..."
            )

            data = self.load_processed_data()

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            ) = self.split_data(data)

            X_train, X_val, X_test = (
                self.transform_data(
                    X_train,
                    X_val,
                    X_test
                )
            )

            X_train, y_train = (
                self.apply_smote(
                    X_train,
                    y_train
                )
            )

            self.save_data(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            )

            self.save_preprocessor()

            logger.info(
                "Data transformation completed"
            )

            return True

        except Exception as e:

            logger.error(
                f"Transformation error: {str(e)}"
            )

            raise e