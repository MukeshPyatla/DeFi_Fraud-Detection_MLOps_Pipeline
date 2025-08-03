"""
Fraud detection model for DeFi transactions.
"""

import joblib
import pickle
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import structlog

from ..utils.logger import LoggerMixin
from ..utils.config import config
from ..utils.validators import DataValidator


class FraudDetector(LoggerMixin):
    """Fraud detection model for DeFi transactions."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize fraud detector.
        
        Args:
            model_path: Path to load existing model from
        """
        super().__init__()
        self.validator = DataValidator()
        
        # Model settings
        self.model_path = model_path or config.get('model.storage.model_path', 'models')
        self.algorithm = config.get('model.parameters.algorithm', 'random_forest')
        self.model_version = None
        self.model_metadata = {}
        
        # Initialize model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Create model directory
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load existing model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.log_info("Fraud detector initialized", 
                     algorithm=self.algorithm,
                     model_path=self.model_path)
    
    def _create_model(self, algorithm: str = None) -> Any:
        """
        Create a new model instance.
        
        Args:
            algorithm: Model algorithm to use
            
        Returns:
            Model instance
        """
        algorithm = algorithm or self.algorithm
        
        if algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=config.get('model.parameters.n_estimators', 100),
                max_depth=config.get('model.parameters.max_depth', 10),
                min_samples_split=config.get('model.parameters.min_samples_split', 2),
                min_samples_leaf=config.get('model.parameters.min_samples_leaf', 1),
                random_state=config.get('model.training.random_state', 42)
            )
        elif algorithm == 'logistic_regression':
            return LogisticRegression(
                random_state=config.get('model.training.random_state', 42),
                max_iter=1000
            )
        elif algorithm == 'svm':
            return SVC(
                probability=True,
                random_state=config.get('model.training.random_state', 42)
            )
        elif algorithm == 'isolation_forest':
            return IsolationForest(
                contamination=0.1,
                random_state=config.get('model.training.random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model training/prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (prepared features, feature names)
        """
        try:
            # Select numeric features
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target variable if present
            if 'risk_score' in numeric_columns:
                numeric_columns.remove('risk_score')
            
            # Remove non-feature columns
            exclude_columns = ['transaction_hash', 'from_address', 'to_address']
            feature_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            # Prepare features
            X = df[feature_columns].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            self.log_info("Features prepared", 
                         original_shape=df.shape,
                         feature_shape=X.shape,
                         feature_count=len(feature_columns))
            
            return X, feature_columns
            
        except Exception as e:
            self.log_error("Failed to prepare features", error=str(e))
            raise
    
    def train_model(self, train_df: pd.DataFrame, 
                   validation_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the fraud detection model.
        
        Args:
            train_df: Training data DataFrame
            validation_df: Optional validation data DataFrame
            
        Returns:
            Training results dictionary
        """
        try:
            self.log_info("Starting model training", 
                         train_shape=train_df.shape,
                         algorithm=self.algorithm)
            
            # Prepare features
            X_train, feature_names = self.prepare_features(train_df)
            self.feature_names = feature_names
            
            # Prepare target variable
            if 'risk_score' in train_df.columns:
                y_train = train_df['risk_score']
                # Convert to binary classification (fraud vs non-fraud)
                y_train_binary = (y_train > 0.5).astype(int)
            else:
                # If no risk score, create synthetic labels based on features
                y_train_binary = self._create_synthetic_labels(X_train)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Create and train model
            self.model = self._create_model()
            
            # Train the model
            self.model.fit(X_train_scaled, y_train_binary)
            
            # Evaluate on training data
            train_predictions = self.model.predict(X_train_scaled)
            train_probabilities = self.model.predict_proba(X_train_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(y_train_binary, train_predictions, train_probabilities)
            
            # Evaluate on validation data if provided
            val_metrics = {}
            if validation_df is not None:
                X_val, _ = self.prepare_features(validation_df)
                X_val_scaled = self.scaler.transform(X_val)
                
                if 'risk_score' in validation_df.columns:
                    y_val = validation_df['risk_score']
                    y_val_binary = (y_val > 0.5).astype(int)
                else:
                    y_val_binary = self._create_synthetic_labels(X_val)
                
                val_predictions = self.model.predict(X_val_scaled)
                val_probabilities = self.model.predict_proba(X_val_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
                
                val_metrics = self._calculate_metrics(y_val_binary, val_predictions, val_probabilities)
            
            # Create model metadata
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_metadata = {
                'algorithm': self.algorithm,
                'version': self.model_version,
                'training_date': datetime.now().isoformat(),
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names),
                'train_samples': len(train_df),
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'model_parameters': self.model.get_params()
            }
            
            self.log_info("Model training completed", 
                         version=self.model_version,
                         train_accuracy=train_metrics.get('accuracy', 0),
                         feature_count=len(self.feature_names))
            
            return {
                'status': 'success',
                'model_version': self.model_version,
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'metadata': self.model_metadata
            }
            
        except Exception as e:
            self.log_error("Model training failed", error=str(e))
            raise
    
    def _create_synthetic_labels(self, X: pd.DataFrame) -> np.ndarray:
        """
        Create synthetic labels for unsupervised learning or when no labels are available.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Synthetic binary labels
        """
        try:
            # Use isolation forest to detect anomalies
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(X)
            
            # Convert to binary labels (1 for anomaly/fraud, 0 for normal)
            labels = (predictions == -1).astype(int)
            
            self.log_info("Synthetic labels created", 
                         total_samples=len(labels),
                         fraud_samples=labels.sum(),
                         fraud_rate=labels.mean())
            
            return labels
            
        except Exception as e:
            self.log_error("Failed to create synthetic labels", error=str(e))
            # Return random labels as fallback
            return np.random.randint(0, 2, size=len(X))
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = (y_true == y_pred).mean()
            metrics['precision'] = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
            metrics['recall'] = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1)
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / max(metrics['precision'] + metrics['recall'], 1e-8)
            
            # AUC if probabilities available
            if y_prob is not None:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            
            # Fraud rate
            metrics['fraud_rate'] = y_true.mean()
            metrics['predicted_fraud_rate'] = y_pred.mean()
            
            return metrics
            
        except Exception as e:
            self.log_error("Failed to calculate metrics", error=str(e))
            return {}
    
    def predict(self, features: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make fraud prediction for transaction features.
        
        Args:
            features: Transaction features (DataFrame or dict)
            
        Returns:
            Prediction results dictionary
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Convert dict to DataFrame if needed
            if isinstance(features, dict):
                features_df = pd.DataFrame([features])
            else:
                features_df = features.copy()
            
            # Prepare features
            X, _ = self.prepare_features(features_df)
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            # Create results
            results = {
                'prediction': int(predictions[0]),
                'is_fraud': bool(predictions[0]),
                'confidence': float(probabilities[0]) if probabilities is not None else 0.5,
                'model_version': self.model_version,
                'prediction_time': datetime.now().isoformat()
            }
            
            self.log_info("Prediction made", 
                         prediction=results['prediction'],
                         confidence=results['confidence'],
                         model_version=self.model_version)
            
            return results
            
        except Exception as e:
            self.log_error("Prediction failed", error=str(e))
            raise
    
    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make batch predictions for multiple transactions.
        
        Args:
            features_df: DataFrame with transaction features
            
        Returns:
            DataFrame with predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Prepare features
            X, _ = self.prepare_features(features_df)
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            # Create results DataFrame
            results_df = features_df.copy()
            results_df['prediction'] = predictions
            results_df['is_fraud'] = predictions.astype(bool)
            results_df['confidence'] = probabilities if probabilities is not None else 0.5
            results_df['model_version'] = self.model_version
            results_df['prediction_time'] = datetime.now().isoformat()
            
            self.log_info("Batch predictions completed", 
                         predictions_count=len(results_df),
                         fraud_count=results_df['is_fraud'].sum())
            
            return results_df
            
        except Exception as e:
            self.log_error("Batch prediction failed", error=str(e))
            raise
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model to file.
        
        Args:
            filepath: Optional filepath. If not provided, generates one.
            
        Returns:
            Path to saved model file
        """
        try:
            if self.model is None:
                raise ValueError("No model to save. Please train the model first.")
            
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.model_path, f"fraud_detector_{timestamp}.joblib")
            
            # Create model package
            model_package = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'algorithm': self.algorithm,
                'version': self.model_version,
                'metadata': self.model_metadata
            }
            
            # Save model
            joblib.dump(model_package, filepath)
            
            self.log_info("Model saved", 
                         filepath=filepath,
                         version=self.model_version)
            
            return filepath
            
        except Exception as e:
            self.log_error("Failed to save model", error=str(e))
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to model file
        """
        try:
            # Load model package
            model_package = joblib.load(filepath)
            
            # Extract components
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.algorithm = model_package['algorithm']
            self.model_version = model_package['version']
            self.model_metadata = model_package['metadata']
            
            self.log_info("Model loaded", 
                         filepath=filepath,
                         version=self.model_version,
                         algorithm=self.algorithm)
            
        except Exception as e:
            self.log_error("Failed to load model", error=str(e))
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {'status': 'no_model_loaded'}
        
        return {
            'algorithm': self.algorithm,
            'version': self.model_version,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'metadata': self.model_metadata
        } 