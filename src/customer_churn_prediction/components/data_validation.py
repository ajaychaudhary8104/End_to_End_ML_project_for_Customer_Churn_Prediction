import os
from src.customer_churn_prediction import logger
import numpy as np
import pandas as pd
from src.customer_churn_prediction.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        
        self.data = pd.read_csv(
            os.path.join(
                self.config.unzip_data_dir,
                "data.csv"
            )
        )
    
    def validate_all_columns_exist(self) -> bool:
        """
        Validate that all required columns exist in the dataset.
        
        Returns:
            bool: True if all columns exist, False otherwise
        """
        try:
            # Read the CSV file from the unzipped directory
            data = self.data
            
            all_cols = list(self.config.all_schema.keys())
            df_cols = list(data.columns)
            
            all_cols_present = True
            missing_columns = []
            
            for col in all_cols:
                if col not in df_cols:
                    all_cols_present = False
                    missing_columns.append(col)
            
            if all_cols_present:
                logger.info(f"✓ All columns are present in the dataset")
            else:
                logger.warning(f"✗ Missing columns: {missing_columns}")
            
            return all_cols_present
            
        except Exception as e:
            logger.error(f"Error validating columns: {str(e)}")
            raise e

    
    def validate_all_rows_exist(self) -> bool:
        """
        Validate that the dataset has data (at least 1 row).
        
        Returns:
            bool: True if data exists, False if dataset is empty
        """
        try:
            data = self.data
            
            if len(data) > 0:
                logger.info(f"✓ Dataset has {len(data)} rows")
                return True
            else:
                logger.warning("✗ Dataset is empty (0 rows)")
                return False
                
        except Exception as e:
            logger.error(f"Error validating rows: {str(e)}")
            raise e

    
    def validate_data_types(self) -> bool:
        """
        Validate that each column has the correct data type.
        
        Returns:
            bool: True if all types match, False if any mismatch
        """
        try:
            data = self.data
            
            schema = self.config.all_schema
            all_types_valid = True
            type_mismatches = []
            
            for col, expected_type in schema.items():
                if col not in data.columns:
                    continue
                    
                actual_type = str(data[col].dtype)
                expected_type_str = expected_type
                
                if actual_type != expected_type_str:
                    all_types_valid = False
                    type_mismatches.append({
                        "column": col,
                        "expected": expected_type_str,
                        "actual": actual_type
                    })
            
            if all_types_valid:
                logger.info(f"✓ All data types match the schema")
            else:
                logger.warning(f"✗ Data type mismatches found:")
                for mismatch in type_mismatches:
                    logger.warning(
                        f"  {mismatch['column']}: expected {mismatch['expected']}, "
                        f"got {mismatch['actual']}"
                    )
            
            return all_types_valid
            
        except Exception as e:
            logger.error(f"Error validating data types: {str(e)}")
            raise e

    
    def validate_missing_values(self) -> bool:
        """
        Validate acceptable levels of missing values in the dataset.
        Allows up to 30% missing values per column.
        
        Returns:
            bool: True if missing values are acceptable, False otherwise
        """
        try:
            data = self.data
            
            missing_threshold = 0.30  # 30% threshold
            all_valid = True
            high_missing_cols = []
            
            for col in data.columns:
                missing_percentage = data[col].isnull().sum() / len(data)
                
                if missing_percentage > missing_threshold:
                    all_valid = False
                    high_missing_cols.append({
                        "column": col,
                        "missing_percentage": round(missing_percentage * 100, 2)
                    })
            
            total_missing = data.isnull().sum().sum()
            total_cells = data.shape[0] * data.shape[1]
            overall_missing_pct = round((total_missing / total_cells) * 100, 2)
            
            if all_valid:
                logger.info(
                    f"✓ Missing values are acceptable "
                    f"(Overall: {overall_missing_pct}% - below {missing_threshold*100}% threshold)"
                )
            else:
                logger.warning(
                    f"✗ Columns with missing values > {missing_threshold*100}%:"
                )
                for col_info in high_missing_cols:
                    logger.warning(
                        f"  {col_info['column']}: {col_info['missing_percentage']}% missing"
                    )
            
            # Log overall missing value statistics
            logger.info(f"Missing values per column:")
            for col in data.columns:
                missing_pct = round((data[col].isnull().sum() / len(data)) * 100, 2)
                if missing_pct > 0:
                    logger.info(f"  {col}: {missing_pct}%")
            
            return all_valid
            
        except Exception as e:
            logger.error(f"Error validating missing values: {str(e)}")
            raise e

    
   
    def validate_numerical_ranges(self) -> bool:
        """
           Validate telecom churn dataset numerical ranges.
        """

        try:
            data = self.data
            validation_issues = []

            # Convert Total Charges
            data["Total Charges"] = pd.to_numeric(
                data["Total Charges"],
                errors="coerce"
            )

            # Tenure Months
            if "Tenure Months" in data.columns:
                min_tenure = data["Tenure Months"].min()
                max_tenure = data["Tenure Months"].max()

                if min_tenure < 0 or max_tenure > 100:
                    validation_issues.append(
                        f"Tenure Months outside valid range "
                        f"(min={min_tenure}, max={max_tenure})"
                    )

            # Monthly Charges
            if "Monthly Charges" in data.columns:
                if (data["Monthly Charges"] < 0).any():
                    validation_issues.append(
                        "Monthly Charges contains negative values"
                    )

            # Total Charges
            if "Total Charges" in data.columns:
                if (data["Total Charges"] < 0).any():
                    validation_issues.append(
                        "Total Charges contains negative values"
                    )

            # Churn Value
            if "Churn Value" in data.columns:
                valid_values = [0, 1]

                invalid = ~data["Churn Value"].isin(valid_values)

                if invalid.any():
                    validation_issues.append(
                        "Churn Value contains invalid values"
                    )

            # Churn Score
            if "Churn Score" in data.columns:
                min_score = data["Churn Score"].min()
                max_score = data["Churn Score"].max()

                if min_score < 0 or max_score > 100:
                    validation_issues.append(
                        f"Churn Score outside 0-100 range "
                        f"(min={min_score}, max={max_score})"
                    )

            # CLTV
            if "CLTV" in data.columns:
                if (data["CLTV"] < 0).any():
                    validation_issues.append(
                        "CLTV contains negative values"
                    )

            # Infinite values
            numeric_cols = data.select_dtypes(
                include=["int64", "float64"]
            ).columns

            for col in numeric_cols:
                if np.isinf(data[col]).any():
                    validation_issues.append(
                        f"{col} contains infinite values"
                    )

            if len(validation_issues) == 0:
                logger.info("✓ All numerical ranges are valid")
                return True

            logger.warning("✗ Numerical validation issues found:")

            for issue in validation_issues:
                logger.warning(f"  {issue}")

            return False

        except Exception as e:
            logger.error(f"Error validating numerical ranges: {str(e)}")
            raise e    

    
    def save_validation_status(self, validation_status: dict) -> None:
        """
        Save validation status to a file for CI/CD integration.
        
        Args:
            validation_status (dict): Dictionary containing validation results
        """
        try:
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            status_content = "DATA VALIDATION REPORT\n"
            status_content += "=" * 50 + "\n\n"
            
            all_passed = True
            for check_name, result in validation_status.items():
                status = "✓ PASSED" if result else "✗ FAILED"
                status_content += f"{check_name}: {status}\n"
                if not result:
                    all_passed = False
            
            status_content += "\n" + "=" * 50 + "\n"
            status_content += f"Overall Status: {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}\n"
            
            with open(self.config.STATUS_FILE, 'w', encoding='utf-8') as f:
                f.write(status_content)
            
            logger.info(f"Validation status saved to: {self.config.STATUS_FILE}")
            
        except Exception as e:
            logger.error(f"Error saving validation status: {str(e)}")
            raise e

    
    def initiate_data_validation(self) -> bool:
        """
        Run all validation checks and save results.
        
        Returns:
            bool: True if all validations pass, False otherwise
        """
        try:
            logger.info("Starting data validation...")
            
            # Run all validation checks
            col_check = self.validate_all_columns_exist()
            row_check = self.validate_all_rows_exist()
            dtype_check = self.validate_data_types()
            missing_check = self.validate_missing_values()
            range_check = self.validate_numerical_ranges()
            
            # Compile results
            validation_results = {
                "Column validation": col_check,
                "Row validation": row_check,
                "Data type validation": dtype_check,
                "Missing values validation": missing_check,
                "Numerical range validation": range_check
            }
            
            # Save status
            self.save_validation_status(validation_results)
            
            # Determine overall result
            all_passed = all(validation_results.values())
            
            if all_passed:
                logger.info("✓ All data validation checks PASSED")
            else:
                logger.warning("✗ Some data validation checks FAILED")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Error during data validation initiation: {str(e)}")
            raise e