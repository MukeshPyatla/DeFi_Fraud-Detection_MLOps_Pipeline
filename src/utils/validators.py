"""
Data validation utilities for the DeFi Fraud Detection Pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
from .logger import LoggerMixin


class DataValidator(LoggerMixin):
    """Data validation utilities for blockchain transaction data."""
    
    def __init__(self):
        super().__init__()
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Set up validation rules for different data types."""
        return {
            'transaction': {
                'required_fields': [
                    'hash', 'from_address', 'to_address', 'value', 
                    'gas_price', 'gas_used', 'block_number', 'timestamp'
                ],
                'field_types': {
                    'hash': str,
                    'from_address': str,
                    'to_address': str,
                    'value': (int, float),
                    'gas_price': (int, float),
                    'gas_used': (int, float),
                    'block_number': int,
                    'timestamp': (int, float)
                },
                'field_constraints': {
                    'hash': {'pattern': r'^0x[a-fA-F0-9]{64}$'},
                    'from_address': {'pattern': r'^0x[a-fA-F0-9]{40}$'},
                    'to_address': {'pattern': r'^0x[a-fA-F0-9]{40}$'},
                    'value': {'min': 0},
                    'gas_price': {'min': 0},
                    'gas_used': {'min': 0},
                    'block_number': {'min': 0},
                    'timestamp': {'min': 0}
                }
            },
            'block': {
                'required_fields': [
                    'number', 'hash', 'timestamp', 'gas_limit', 'gas_used'
                ],
                'field_types': {
                    'number': int,
                    'hash': str,
                    'timestamp': (int, float),
                    'gas_limit': (int, float),
                    'gas_used': (int, float)
                },
                'field_constraints': {
                    'number': {'min': 0},
                    'hash': {'pattern': r'^0x[a-fA-F0-9]{64}$'},
                    'timestamp': {'min': 0},
                    'gas_limit': {'min': 0},
                    'gas_used': {'min': 0}
                }
            }
        }
    
    def validate_transaction_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate transaction data against defined rules.
        
        Args:
            data: Transaction data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        rules = self.validation_rules['transaction']
        
        # Check required fields
        for field in rules['required_fields']:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in rules['field_types'].items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    errors.append(f"Invalid type for {field}: expected {expected_type}, got {type(data[field])}")
        
        # Check field constraints
        for field, constraints in rules['field_constraints'].items():
            if field in data and data[field] is not None:
                for constraint_type, constraint_value in constraints.items():
                    if constraint_type == 'pattern':
                        if not re.match(constraint_value, str(data[field])):
                            errors.append(f"Invalid format for {field}: does not match pattern {constraint_value}")
                    elif constraint_type == 'min':
                        if data[field] < constraint_value:
                            errors.append(f"Invalid value for {field}: must be >= {constraint_value}")
                    elif constraint_type == 'max':
                        if data[field] > constraint_value:
                            errors.append(f"Invalid value for {field}: must be <= {constraint_value}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.log_warning("Transaction validation failed", errors=errors, data=data)
        else:
            self.log_debug("Transaction validation passed", data=data)
        
        return is_valid, errors
    
    def validate_block_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate block data against defined rules.
        
        Args:
            data: Block data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        rules = self.validation_rules['block']
        
        # Check required fields
        for field in rules['required_fields']:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in rules['field_types'].items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    errors.append(f"Invalid type for {field}: expected {expected_type}, got {type(data[field])}")
        
        # Check field constraints
        for field, constraints in rules['field_constraints'].items():
            if field in data and data[field] is not None:
                for constraint_type, constraint_value in constraints.items():
                    if constraint_type == 'pattern':
                        if not re.match(constraint_value, str(data[field])):
                            errors.append(f"Invalid format for {field}: does not match pattern {constraint_value}")
                    elif constraint_type == 'min':
                        if data[field] < constraint_value:
                            errors.append(f"Invalid value for {field}: must be >= {constraint_value}")
                    elif constraint_type == 'max':
                        if data[field] > constraint_value:
                            errors.append(f"Invalid value for {field}: must be <= {constraint_value}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.log_warning("Block validation failed", errors=errors, data=data)
        else:
            self.log_debug("Block validation passed", data=data)
        
        return is_valid, errors
    
    def validate_dataframe(self, df: pd.DataFrame, expected_columns: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate pandas DataFrame for data quality issues.
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check for expected columns
        if expected_columns:
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                errors.append(f"Missing expected columns: {missing_columns}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if not columns_with_nulls.empty:
            errors.append(f"Columns with null values: {columns_with_nulls.to_dict()}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            errors.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
                errors.append(f"Column {col} has {infinite_count} infinite values")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.log_warning("DataFrame validation failed", errors=errors, shape=df.shape)
        else:
            self.log_debug("DataFrame validation passed", shape=df.shape)
        
        return is_valid, errors
    
    def validate_feature_data(self, features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate feature data for model training/inference.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for required features
        required_features = [
            'transaction_amount', 'gas_price', 'gas_used', 'block_time',
            'sender_balance', 'receiver_balance', 'transaction_frequency'
        ]
        
        for feature in required_features:
            if feature not in features:
                errors.append(f"Missing required feature: {feature}")
            elif features[feature] is None:
                errors.append(f"Feature {feature} is None")
            elif isinstance(features[feature], (int, float)) and np.isnan(features[feature]):
                errors.append(f"Feature {feature} is NaN")
            elif isinstance(features[feature], (int, float)) and np.isinf(features[feature]):
                errors.append(f"Feature {feature} is infinite")
        
        # Check feature value ranges
        feature_ranges = {
            'transaction_amount': (0, float('inf')),
            'gas_price': (0, float('inf')),
            'gas_used': (0, float('inf')),
            'block_time': (0, float('inf')),
            'sender_balance': (0, float('inf')),
            'receiver_balance': (0, float('inf')),
            'transaction_frequency': (0, float('inf'))
        }
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in features and features[feature] is not None:
                value = features[feature]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        errors.append(f"Feature {feature} value {value} outside valid range [{min_val}, {max_val}]")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.log_warning("Feature validation failed", errors=errors, features=features)
        else:
            self.log_debug("Feature validation passed", features=features)
        
        return is_valid, errors
    
    def detect_anomalies(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect anomalies in numerical data using statistical methods.
        
        Args:
            data: DataFrame containing numerical data
            columns: List of columns to check for anomalies
            
        Returns:
            DataFrame with anomaly flags
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        anomaly_df = data.copy()
        
        for column in columns:
            if column in data.columns:
                # Calculate z-score
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                anomaly_df[f'{column}_anomaly'] = z_scores > 3
                
                # Calculate IQR-based outliers
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomaly_df[f'{column}_outlier'] = (data[column] < lower_bound) | (data[column] > upper_bound)
        
        self.log_info("Anomaly detection completed", columns_checked=columns)
        return anomaly_df
    
    def generate_validation_report(self, validation_results: List[Tuple[bool, List[str]]]) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Validation report dictionary
        """
        total_validations = len(validation_results)
        passed_validations = sum(1 for is_valid, _ in validation_results if is_valid)
        failed_validations = total_validations - passed_validations
        
        all_errors = []
        for _, errors in validation_results:
            all_errors.extend(errors)
        
        report = {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': failed_validations,
            'success_rate': passed_validations / total_validations if total_validations > 0 else 0,
            'total_errors': len(all_errors),
            'unique_errors': list(set(all_errors)),
            'error_summary': {}
        }
        
        # Count error occurrences
        for error in all_errors:
            report['error_summary'][error] = all_errors.count(error)
        
        self.log_info("Validation report generated", report=report)
        return report 