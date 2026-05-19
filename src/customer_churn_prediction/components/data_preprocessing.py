import os
import pandas as pd
from src.customer_churn_prediction import logger
from src.customer_churn_prediction.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig):

        self.config = config

    # ======================================================
    # Raw File Path
    # ======================================================

    def get_raw_file_path(self):

        return os.path.join(
            self.config.raw_data_dir,
            self.config.input_file_name
        )

    # ======================================================
    # Load Data
    # ======================================================

    def load_data(self):

        raw_file_path = self.get_raw_file_path()

        if not os.path.exists(raw_file_path):

            raise FileNotFoundError(
                f"File not found: {raw_file_path}"
            )

        logger.info(
            f"Loading data from: {raw_file_path}"
        )

        data = pd.read_csv(raw_file_path)

        logger.info(
            f"Data shape: {data.shape}"
        )

        return data

    # ======================================================
    # Drop Columns
    # ======================================================

    def drop_columns(self, data):

        drop_cols = [
            col for col in self.config.drop_columns
            if col in data.columns
        ]

        if drop_cols:

            logger.info(
                f"Dropping columns: {drop_cols}"
            )

            data = data.drop(
                columns=drop_cols
            )

        return data

    # ======================================================
    # Feature Engineering
    # ======================================================

    def create_derived_features(self, data):

        logger.info(
            "Creating derived features..."
        )

        # Convert Total Charges
        if "Total Charges" in data.columns:

            data["Total Charges"] = pd.to_numeric(
                data["Total Charges"],
                errors="coerce"
            )

        # Avg Monthly Spend
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

        # Long Term Customer
        if "Tenure Months" in data.columns:

            data["LongTermCustomer"] = (
                data["Tenure Months"] >= 24
            ).astype(int)

        # High Monthly Charges
        if "Monthly Charges" in data.columns:

            threshold = data[
                "Monthly Charges"
            ].median()

            data["HighMonthlyCharges"] = (
                data["Monthly Charges"] > threshold
            ).astype(int)

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

        existing_cols = [
            col for col in service_cols
            if col in data.columns
        ]

        if existing_cols:

            data["TotalServices"] = data[
                existing_cols
            ].apply(
                lambda row: sum(row == "Yes"),
                axis=1
            )

        logger.info(
            f"Feature engineering completed: "
            f"{data.shape}"
        )

        return data

    # ======================================================
    # Save Cleaned Data
    # ======================================================

    def save_cleaned_data(self, data):

        os.makedirs(
            self.config.root_dir,
            exist_ok=True
        )

        output_path = (
            self.config.processed_data_file
        )

        data.to_csv(
            output_path,
            index=False
        )

        logger.info(
            f"Cleaned data saved to: "
            f"{output_path}"
        )

    # ======================================================
    # Main Pipeline
    # ======================================================

    def initiate_data_preprocessing(self):

        try:

            logger.info(
                "Starting data preprocessing..."
            )

            data = self.load_data()

            data = self.drop_columns(data)

            data = self.create_derived_features(data)

            self.save_cleaned_data(data)

            logger.info(
                "Data preprocessing completed"
            )

            return True

        except Exception as e:

            logger.error(
                f"Preprocessing error: {str(e)}"
            )

            raise e