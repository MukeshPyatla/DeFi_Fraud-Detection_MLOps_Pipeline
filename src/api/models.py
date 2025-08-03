"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class ModelStatus(str, Enum):
    """Model status enumeration."""
    LOADED = "loaded"
    NOT_LOADED = "not_loaded"
    TRAINING = "training"
    ERROR = "error"


class PredictionRequest(BaseModel):
    """Request model for fraud prediction."""
    
    transaction_hash: str = Field(..., description="Transaction hash")
    from_address: str = Field(..., description="Sender address")
    to_address: str = Field(..., description="Receiver address")
    value: float = Field(..., description="Transaction value in ETH")
    gas_price: float = Field(..., description="Gas price in Gwei")
    gas_used: float = Field(..., description="Gas used")
    block_number: int = Field(..., description="Block number")
    timestamp: int = Field(..., description="Block timestamp")
    
    # Optional fields
    nonce: Optional[int] = Field(None, description="Transaction nonce")
    input_data: Optional[str] = Field(None, description="Transaction input data")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "from_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                "to_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                "value": 0.1,
                "gas_price": 20.5,
                "gas_used": 21000,
                "block_number": 15000000,
                "timestamp": 1640995200,
                "nonce": 0,
                "input_data": "0x"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    
    prediction: int = Field(..., description="Prediction (0=normal, 1=fraud)")
    is_fraud: bool = Field(..., description="Whether transaction is flagged as fraud")
    confidence: float = Field(..., description="Prediction confidence score")
    model_version: str = Field(..., description="Model version used")
    prediction_time: str = Field(..., description="Prediction timestamp")
    features_used: List[str] = Field(..., description="Features used for prediction")
    risk_score: float = Field(..., description="Calculated risk score")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "is_fraud": False,
                "confidence": 0.85,
                "model_version": "20231201_143022",
                "prediction_time": "2023-12-01T14:30:22.123456",
                "features_used": ["transaction_amount", "gas_price", "gas_used"],
                "risk_score": 0.15
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch fraud prediction."""
    
    transactions: List[PredictionRequest] = Field(..., description="List of transactions to predict")
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                        "from_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                        "to_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                        "value": 0.1,
                        "gas_price": 20.5,
                        "gas_used": 21000,
                        "block_number": 15000000,
                        "timestamp": 1640995200
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch fraud prediction."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_transactions: int = Field(..., description="Total number of transactions processed")
    fraud_count: int = Field(..., description="Number of transactions flagged as fraud")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 0,
                        "is_fraud": False,
                        "confidence": 0.85,
                        "model_version": "20231201_143022",
                        "prediction_time": "2023-12-01T14:30:22.123456",
                        "features_used": ["transaction_amount", "gas_price", "gas_used"],
                        "risk_score": 0.15
                    }
                ],
                "total_transactions": 1,
                "fraud_count": 0,
                "processing_time": 0.123
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    model_status: ModelStatus = Field(..., description="Model status")
    model_version: Optional[str] = Field(None, description="Current model version")
    uptime: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_status": "loaded",
                "model_version": "20231201_143022",
                "uptime": 3600.5,
                "timestamp": "2023-12-01T14:30:22.123456"
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    algorithm: str = Field(..., description="Model algorithm")
    version: str = Field(..., description="Model version")
    feature_count: int = Field(..., description="Number of features")
    feature_names: List[str] = Field(..., description="List of feature names")
    training_date: str = Field(..., description="Model training date")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm": "random_forest",
                "version": "20231201_143022",
                "feature_count": 15,
                "feature_names": ["transaction_amount", "gas_price", "gas_used"],
                "training_date": "2023-12-01T14:30:22.123456",
                "performance_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90,
                    "auc": 0.94
                }
            }
        }


class TrainingRequest(BaseModel):
    """Request model for model training."""
    
    data_path: str = Field(..., description="Path to training data")
    algorithm: Optional[str] = Field(None, description="Model algorithm to use")
    test_size: float = Field(0.2, description="Test set size")
    random_state: int = Field(42, description="Random seed")
    
    class Config:
        schema_extra = {
            "example": {
                "data_path": "data/processed/train_features.csv",
                "algorithm": "random_forest",
                "test_size": 0.2,
                "random_state": 42
            }
        }


class TrainingResponse(BaseModel):
    """Response model for model training."""
    
    status: str = Field(..., description="Training status")
    model_version: str = Field(..., description="New model version")
    training_time: float = Field(..., description="Training time in seconds")
    train_metrics: Dict[str, float] = Field(..., description="Training metrics")
    validation_metrics: Dict[str, float] = Field(..., description="Validation metrics")
    model_path: str = Field(..., description="Path to saved model")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "model_version": "20231201_143022",
                "training_time": 45.2,
                "train_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90
                },
                "validation_metrics": {
                    "accuracy": 0.94,
                    "precision": 0.91,
                    "recall": 0.87,
                    "f1_score": 0.89
                },
                "model_path": "models/fraud_detector_20231201_143022.joblib"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Model not loaded",
                "detail": "Please train or load a model first",
                "timestamp": "2023-12-01T14:30:22.123456"
            }
        } 