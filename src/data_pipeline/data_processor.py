"""
Data processor for handling data storage, validation, and preprocessing.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle

from ..utils.logger import LoggerMixin
from ..utils.config import config
from ..utils.validators import DataValidator
from ..utils.encryption import DataEncryption


class DataProcessor(LoggerMixin):
    """Data processor for handling data storage, validation, and preprocessing."""
    
    def __init__(self):
        """Initialize data processor."""
        super().__init__()
        self.validator = DataValidator()
        self.encryption = DataEncryption()
        
        # Storage paths
        self.raw_data_path = config.get('data_pipeline.storage.raw_data_path', 'data/raw')
        self.processed_data_path = config.get('data_pipeline.storage.processed_data_path', 'data/processed')
        
        # Create directories if they don't exist
        Path(self.raw_data_path).mkdir(parents=True, exist_ok=True)
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
        
        # Processing settings
        self.backup_enabled = config.get('data_pipeline.storage.backup_enabled', True)
        self.compression = config.get('data_pipeline.storage.compression', 'gzip')
        
        self.log_info("Data processor initialized", 
                     raw_path=self.raw_data_path,
                     processed_path=self.processed_data_path)
    
    def save_raw_data(self, data: List[Dict[str, Any]], 
                     filename: Optional[str] = None) -> str:
        """
        Save raw transaction data to file.
        
        Args:
            data: List of transaction dictionaries
            filename: Optional filename. If not provided, generates one.
            
        Returns:
            Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"raw_transactions_{timestamp}.json"
            
            filepath = os.path.join(self.raw_data_path, filename)
            
            # Secure the data before saving
            secured_data = []
            for tx in data:
                secured_tx = self.encryption.secure_transaction_data(tx)
                secured_data.append(secured_tx)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(secured_data, f, indent=2)
            
            self.log_info("Raw data saved", 
                         filepath=filepath, 
                         transaction_count=len(data))
            
            return filepath
            
        except Exception as e:
            self.log_error("Failed to save raw data", 
                          error=str(e), 
                          filename=filename)
            raise
    
    def load_raw_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load raw transaction data from file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            List of transaction dictionaries
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.log_info("Raw data loaded", 
                         filepath=filepath, 
                         transaction_count=len(data))
            
            return data
            
        except Exception as e:
            self.log_error("Failed to load raw data", 
                          error=str(e), 
                          filepath=filepath)
            raise
    
    def save_processed_data(self, df: pd.DataFrame, 
                          filename: Optional[str] = None) -> str:
        """
        Save processed feature data to file.
        
        Args:
            df: Feature DataFrame
            filename: Optional filename. If not provided, generates one.
            
        Returns:
            Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_features_{timestamp}.csv"
            
            filepath = os.path.join(self.processed_data_path, filename)
            
            # Validate DataFrame before saving
            is_valid, errors = self.validator.validate_dataframe(df)
            if not is_valid:
                self.log_warning("DataFrame validation failed before saving", errors=errors)
            
            # Save with compression if enabled
            if self.compression == 'gzip':
                filepath = filepath.replace('.csv', '.csv.gz')
                df.to_csv(filepath, index=False, compression='gzip')
            else:
                df.to_csv(filepath, index=False)
            
            self.log_info("Processed data saved", 
                         filepath=filepath, 
                         shape=df.shape)
            
            return filepath
            
        except Exception as e:
            self.log_error("Failed to save processed data", 
                          error=str(e), 
                          filename=filename)
            raise
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """
        Load processed feature data from file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Feature DataFrame
        """
        try:
            if filepath.endswith('.gz'):
                df = pd.read_csv(filepath, compression='gzip')
            else:
                df = pd.read_csv(filepath)
            
            # Validate loaded DataFrame
            is_valid, errors = self.validator.validate_dataframe(df)
            if not is_valid:
                self.log_warning("Loaded DataFrame validation failed", errors=errors)
            
            self.log_info("Processed data loaded", 
                         filepath=filepath, 
                         shape=df.shape)
            
            return df
            
        except Exception as e:
            self.log_error("Failed to load processed data", 
                          error=str(e), 
                          filepath=filepath)
            raise
    
    def save_metadata(self, metadata: Dict[str, Any], 
                     filename: Optional[str] = None) -> str:
        """
        Save processing metadata to file.
        
        Args:
            metadata: Metadata dictionary
            filename: Optional filename. If not provided, generates one.
            
        Returns:
            Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metadata_{timestamp}.json"
            
            filepath = os.path.join(self.processed_data_path, filename)
            
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.log_info("Metadata saved", 
                         filepath=filepath, 
                         metadata_keys=list(metadata.keys()))
            
            return filepath
            
        except Exception as e:
            self.log_error("Failed to save metadata", 
                          error=str(e), 
                          filename=filename)
            raise
    
    def load_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Load processing metadata from file.
        
        Args:
            filepath: Path to the metadata file
            
        Returns:
            Metadata dictionary
        """
        try:
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            
            self.log_info("Metadata loaded", 
                         filepath=filepath, 
                         metadata_keys=list(metadata.keys()))
            
            return metadata
            
        except Exception as e:
            self.log_error("Failed to load metadata", 
                          error=str(e), 
                          filepath=filepath)
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess feature data for model training.
        
        Args:
            df: Raw feature DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            processed_df = df.copy()
            
            # Handle missing values
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if processed_df[col].isnull().sum() > 0:
                    # Fill with median for numeric columns
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
            
            # Handle infinite values
            for col in numeric_columns:
                processed_df[col] = processed_df[col].replace([np.inf, -np.inf], np.nan)
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
            
            # Remove duplicate rows
            initial_rows = len(processed_df)
            processed_df = processed_df.drop_duplicates()
            if len(processed_df) < initial_rows:
                self.log_info("Removed duplicate rows", 
                             removed_count=initial_rows - len(processed_df))
            
            # Feature scaling (optional - can be done during model training)
            # For now, we'll just ensure all features are numeric
            
            # Remove non-numeric columns for model training
            exclude_columns = ['transaction_hash', 'from_address', 'to_address']
            model_columns = [col for col in processed_df.columns if col not in exclude_columns]
            processed_df = processed_df[model_columns]
            
            self.log_info("Data preprocessing completed", 
                         initial_shape=df.shape,
                         final_shape=processed_df.shape)
            
            return processed_df
            
        except Exception as e:
            self.log_error("Failed to preprocess data", error=str(e))
            raise
    
    def create_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split for model training.
        
        Args:
            df: Feature DataFrame
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        try:
            # Use sklearn's train_test_split
            from sklearn.model_selection import train_test_split
            
            # Separate features and target
            feature_columns = [col for col in df.columns if col != 'risk_score']
            X = df[feature_columns]
            y = df['risk_score'] if 'risk_score' in df.columns else None
            
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Reconstruct DataFrames
                train_df = X_train.copy()
                train_df['risk_score'] = y_train
                
                test_df = X_test.copy()
                test_df['risk_score'] = y_test
            else:
                # No target variable, just split features
                train_df, test_df = train_test_split(
                    df, test_size=test_size, random_state=random_state
                )
            
            self.log_info("Train/test split created", 
                         train_shape=train_df.shape,
                         test_shape=test_df.shape,
                         test_size=test_size)
            
            return train_df, test_df
            
        except Exception as e:
            self.log_error("Failed to create train/test split", error=str(e))
            raise
    
    def backup_data(self, source_path: str, backup_dir: str = None) -> str:
        """
        Create a backup of data file.
        
        Args:
            source_path: Path to source file
            backup_dir: Backup directory. If None, uses default.
            
        Returns:
            Path to backup file
        """
        try:
            if not self.backup_enabled:
                return source_path
            
            if backup_dir is None:
                backup_dir = os.path.join(self.processed_data_path, 'backups')
            
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(source_path)
            name, ext = os.path.splitext(filename)
            backup_filename = f"{name}_backup_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy file
            import shutil
            shutil.copy2(source_path, backup_path)
            
            self.log_info("Data backup created", 
                         source=source_path,
                         backup=backup_path)
            
            return backup_path
            
        except Exception as e:
            self.log_error("Failed to create backup", 
                          error=str(e), 
                          source_path=source_path)
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Summary dictionary
        """
        try:
            summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_summary': {},
                'categorical_summary': {}
            }
            
            # Numeric columns summary
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                summary['numeric_summary'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                }
            
            # Categorical columns summary
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                summary['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None
                }
            
            self.log_info("Data summary generated", 
                         shape=df.shape,
                         numeric_columns=len(numeric_columns),
                         categorical_columns=len(categorical_columns))
            
            return summary
            
        except Exception as e:
            self.log_error("Failed to generate data summary", error=str(e))
            return {}
    
    def cleanup_old_files(self, directory: str, days_old: int = 30) -> int:
        """
        Clean up old files in a directory.
        
        Args:
            directory: Directory to clean
            days_old: Files older than this many days will be deleted
            
        Returns:
            Number of files deleted
        """
        try:
            if not os.path.exists(directory):
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_time < cutoff_date:
                        os.remove(filepath)
                        deleted_count += 1
            
            if deleted_count > 0:
                self.log_info("Cleaned up old files", 
                             directory=directory,
                             deleted_count=deleted_count,
                             days_old=days_old)
            
            return deleted_count
            
        except Exception as e:
            self.log_error("Failed to cleanup old files", 
                          error=str(e), 
                          directory=directory)
            return 0 