import os
import pickle
import pandas as pd
import numpy as np
from src.customer_churn_prediction import logger
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)
from sklearn.impute import SimpleImputer
from src.customer_churn_prediction.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig):

        self.config = config

        self.preprocessor = None

    # ======================================================
    # Load Data
    # ======================================================

    def get_raw_file_path(self) -> str:

        return os.path.join(
            self.config.raw_data_dir,
            self.config.input_file_name
        )

    def load_data(self) -> pd.DataFrame:

        raw_file_path = self.get_raw_file_path()

        if not os.path.exists(raw_file_path):

            raise FileNotFoundError(
                f"Raw data file not found: {raw_file_path}"
            )

        logger.info(
            f"Loading raw data from: {raw_file_path}"
        )

        data = pd.read_csv(raw_file_path)

        logger.info(
            f"Dataset loaded successfully: {data.shape}"
        )

        return data

    # ======================================================
    # Drop Unnecessary Columns
    # ======================================================

    def drop_columns(self,data: pd.DataFrame) -> pd.DataFrame:

        drop_cols = [
            col for col in self.config.drop_columns
            if col in data.columns
        ]

        if drop_cols:

            logger.info(
                f"Dropping columns: {drop_cols}"
            )

            data = data.drop(columns=drop_cols)

        return data

    # ======================================================
    # Create Derived Features
    # ======================================================

    def create_derived_features(self,data: pd.DataFrame) -> pd.DataFrame:

        logger.info(
            "Creating derived features..."
        )

        # Convert Total Charges
        if "Total Charges" in data.columns:

            data["Total Charges"] = pd.to_numeric(
                data["Total Charges"],
                errors="coerce"
            )

        # Average Monthly Spend
        if all(
            col in data.columns
            for col in [
                "Total Charges",
                "Tenure Months"
            ]
        ):

            data["Avg Monthly Spend"] = (
                data["Total Charges"] /
                (data["Tenure Months"] + 1)
            )

            logger.info(
                "Created feature: Avg Monthly Spend"
            )

        # Long-Term Customer
        if "Tenure Months" in data.columns:

            data["LongTermCustomer"] = (
                data["Tenure Months"] >= 24
            ).astype(int)

            logger.info(
                "Created feature: LongTermCustomer"
            )

        # High Monthly Charges
        if "Monthly Charges" in data.columns:

            threshold = data[
                "Monthly Charges"
            ].median()

            data["HighMonthlyCharges"] = (
                data["Monthly Charges"] > threshold
            ).astype(int)

            logger.info(
                "Created feature: HighMonthlyCharges"
            )

        # Total Services
        service_cols = [
            "Phone Service",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies"
        ]

        existing_service_cols = [
            col for col in service_cols
            if col in data.columns
        ]

        if existing_service_cols:

            data["TotalServices"] = data[
                existing_service_cols
            ].apply(
                lambda row: sum(row == "Yes"),
                axis=1
            )

            logger.info(
                "Created feature: TotalServices"
            )

        logger.info(
            f"Feature engineering completed: {data.shape}"
        )

        return data

    # ======================================================
    # Build Preprocessor
    # ======================================================

    def build_preprocessor(self,numeric_columns,categorical_columns) -> ColumnTransformer:

        logger.info(
            "Building preprocessing pipeline..."
        )

        # Numeric Pipeline
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

        # Categorical Pipeline
        categorical_pipeline = Pipeline(
            steps=[(
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",sparse_output= False
                    )
                )
            ]
        )

        # Column Transformer
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
            ]
        )

        logger.info(
            "Preprocessing pipeline built successfully"
        )

        return preprocessor

    # ======================================================
    # Save Preprocessor
    # ======================================================

    def save_preprocessor(self) -> None:

        preprocessor_path = os.path.join(
            self.config.root_dir,
            "preprocessor.pkl"
        )

        os.makedirs(
            self.config.root_dir,
            exist_ok=True
        )

        with open(preprocessor_path, "wb") as file:

            pickle.dump(
                self.preprocessor,
                file
            )

        logger.info(
            f"Preprocessor saved at: {preprocessor_path}"
        )

    # ======================================================
    # Save Processed Data
    # ======================================================

    def save_processed_data(self, transformed_data, feature_names) -> None:

        processed_df = pd.DataFrame(
            transformed_data,
            columns=feature_names
        )

        os.makedirs(
            self.config.root_dir,
            exist_ok=True
        )

        processed_df.to_csv(
            self.config.processed_data_file,
            index=False
        )

        logger.info(
            f"Processed data saved to: "
            f"{self.config.processed_data_file}"
        )

        logger.info(
            f"Processed data shape: "
            f"{processed_df.shape}"
        )

    # ======================================================
    # Main Preprocessing Pipeline
    # ======================================================

    def initiate_data_preprocessing(self) -> bool:

        try:

            logger.info(
                "Starting data preprocessing..."
            )

            # Load Data
            data = self.load_data()

            # Drop Columns
            data = self.drop_columns(data)

            # Feature Engineering
            data = self.create_derived_features(data)

            # Separate Target
            target_column = self.config.target_column

            y = data[target_column]

            X = data.drop(columns=[target_column])

            # Numeric Columns
            numeric_columns = [
                col for col in self.config.numeric_columns
                if col in X.columns
            ]

            # Add Derived Numeric Features
            derived_numeric = [
                "Avg Monthly Spend",
                "TotalServices"
            ]

            numeric_columns += [
                col for col in derived_numeric
                if col in X.columns
                and col not in numeric_columns
            ]

            # Categorical Columns
            categorical_columns = [
                col for col in self.config.categorical_columns
                if col in X.columns
            ]

            # Add Derived Categorical Features
            derived_categorical = [
                "LongTermCustomer",
                "HighMonthlyCharges"
            ]

            categorical_columns += [
                col for col in derived_categorical
                if col in X.columns
                and col not in categorical_columns
            ]

            logger.info(
                f"Numeric columns: {numeric_columns}"
            )

            logger.info(
                f"Categorical columns: {categorical_columns}"
            )

            # Build Preprocessor
            self.preprocessor = self.build_preprocessor(
                numeric_columns,
                categorical_columns
            )

            # Fit + Transform
            transformed_X = self.preprocessor.fit_transform(X)

            # Get Feature Names
            feature_names = (
                self.preprocessor.get_feature_names_out()
            )


            # Final DataFrame
            processed_df = pd.DataFrame(
                transformed_X,
                columns=feature_names
            )

            processed_df[target_column] = y.values

            # Save Processed Data
            self.save_processed_data(
                processed_df,
                processed_df.columns
            )

            # Save Preprocessor
            self.save_preprocessor()

            logger.info(
                "Data preprocessing completed successfully"
            )

            return True

        except Exception as e:

            logger.error(
                f"Error during preprocessing: {str(e)}"
            )

            raise e