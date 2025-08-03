"""
API endpoints for DeFi fraud detection.
"""

import time
from typing import Dict, List, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from .models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse, TrainingRequest, TrainingResponse, ErrorResponse,
    ModelStatus
)
from ..model.fraud_detector import FraudDetector
from ..data_pipeline.feature_engineer import FeatureEngineer
from ..utils.logger import setup_logger
from ..utils.config import config
from ..utils.audit_logger import audit_logger

# Initialize components
logger = setup_logger("api")
fraud_detector = FraudDetector()
feature_engineer = FeatureEngineer()

# Service start time for uptime calculation
start_time = datetime.now()

# Create router
router = APIRouter()


def get_fraud_detector() -> FraudDetector:
    """Dependency to get fraud detector instance."""
    return fraud_detector


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        uptime = (datetime.now() - start_time).total_seconds()
        
        # Check model status
        model_info = fraud_detector.get_model_info()
        if model_info.get('status') == 'no_model_loaded':
            model_status = ModelStatus.NOT_LOADED
            model_version = None
        else:
            model_status = ModelStatus.LOADED
            model_version = model_info.get('version')
        
        return HealthResponse(
            status="healthy",
            model_status=model_status,
            model_version=model_version,
            uptime=uptime,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed")


@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    request: PredictionRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Predict fraud for a single transaction."""
    try:
        # Convert request to transaction format
        transaction = {
            'hash': request.transaction_hash,
            'from_address': request.from_address,
            'to_address': request.to_address,
            'value': int(request.value * 1e18),  # Convert to Wei
            'gas_price': int(request.gas_price * 1e9),  # Convert to Wei
            'gas_used': request.gas_used,
            'block_number': request.block_number,
            'timestamp': request.timestamp,
            'nonce': request.nonce or 0,
            'input': request.input_data or '0x'
        }
        
        # Engineer features
        features = feature_engineer.create_fraud_features(transaction, {}, {})
        
        if not features:
            raise HTTPException(status_code=400, detail="Failed to create features")
        
        # Make prediction
        prediction_result = detector.predict(features)
        
        # Log prediction for audit
        audit_logger.log_prediction(
            prediction_id=request.transaction_hash,
            model_version=prediction_result['model_version'],
            input_data=transaction,
            prediction=prediction_result,
            confidence=prediction_result['confidence']
        )
        
        # Create response
        response = PredictionResponse(
            prediction=prediction_result['prediction'],
            is_fraud=prediction_result['is_fraud'],
            confidence=prediction_result['confidence'],
            model_version=prediction_result['model_version'],
            prediction_time=prediction_result['prediction_time'],
            features_used=list(features.keys()),
            risk_score=features.get('risk_score', 0.0)
        )
        
        logger.info("Prediction completed", 
                   tx_hash=request.transaction_hash,
                   prediction=response.prediction,
                   confidence=response.confidence)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed", error=str(e), tx_hash=request.transaction_hash)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    request: BatchPredictionRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Predict fraud for multiple transactions."""
    try:
        start_time = time.time()
        
        # Convert requests to transactions
        transactions = []
        for tx_request in request.transactions:
            transaction = {
                'hash': tx_request.transaction_hash,
                'from_address': tx_request.from_address,
                'to_address': tx_request.to_address,
                'value': int(tx_request.value * 1e18),
                'gas_price': int(tx_request.gas_price * 1e9),
                'gas_used': tx_request.gas_used,
                'block_number': tx_request.block_number,
                'timestamp': tx_request.timestamp,
                'nonce': tx_request.nonce or 0,
                'input': tx_request.input_data or '0x'
            }
            transactions.append(transaction)
        
        # Engineer features for all transactions
        features_list = []
        for transaction in transactions:
            features = feature_engineer.create_fraud_features(transaction, {}, {})
            if features:
                features['transaction_hash'] = transaction['hash']
                features_list.append(features)
        
        if not features_list:
            raise HTTPException(status_code=400, detail="Failed to create features for any transaction")
        
        # Convert to DataFrame for batch prediction
        import pandas as pd
        features_df = pd.DataFrame(features_list)
        
        # Make batch predictions
        predictions_df = detector.predict_batch(features_df)
        
        # Convert to response format
        predictions = []
        fraud_count = 0
        
        for _, row in predictions_df.iterrows():
            prediction = PredictionResponse(
                prediction=int(row['prediction']),
                is_fraud=bool(row['is_fraud']),
                confidence=float(row['confidence']),
                model_version=str(row['model_version']),
                prediction_time=str(row['prediction_time']),
                features_used=[col for col in features_df.columns if col not in ['transaction_hash', 'prediction', 'is_fraud', 'confidence', 'model_version', 'prediction_time']],
                risk_score=float(row.get('risk_score', 0.0))
            )
            predictions.append(prediction)
            
            if prediction.is_fraud:
                fraud_count += 1
        
        processing_time = time.time() - start_time
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(predictions),
            fraud_count=fraud_count,
            processing_time=processing_time
        )
        
        logger.info("Batch prediction completed", 
                   total_transactions=response.total_transactions,
                   fraud_count=response.fraud_count,
                   processing_time=response.processing_time)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(detector: FraudDetector = Depends(get_fraud_detector)):
    """Get information about the current model."""
    try:
        model_info = detector.get_model_info()
        
        if model_info.get('status') == 'no_model_loaded':
            raise HTTPException(status_code=404, detail="No model loaded")
        
        # Get performance metrics from metadata
        metadata = model_info.get('metadata', {})
        train_metrics = metadata.get('train_metrics', {})
        
        response = ModelInfoResponse(
            algorithm=model_info['algorithm'],
            version=model_info['version'],
            feature_count=model_info['feature_count'],
            feature_names=model_info['feature_names'],
            training_date=metadata.get('training_date', ''),
            performance_metrics=train_metrics
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/model/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    detector: FraudDetector = Depends(get_fraud_detector)
):
    """Train a new model."""
    try:
        import pandas as pd
        
        # Load training data
        train_df = pd.read_csv(request.data_path)
        
        # Create train/test split
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            train_df, 
            test_size=request.test_size, 
            random_state=request.random_state
        )
        
        # Train model
        start_time = time.time()
        training_result = detector.train_model(train_data, test_data)
        training_time = time.time() - start_time
        
        # Save model
        model_path = detector.save_model()
        
        # Log model change for audit
        old_version = detector.model_version
        audit_logger.log_model_change(
            old_version=old_version or "none",
            new_version=training_result['model_version'],
            change_reason="Manual training via API",
            performance_metrics=training_result['train_metrics']
        )
        
        response = TrainingResponse(
            status=training_result['status'],
            model_version=training_result['model_version'],
            training_time=training_time,
            train_metrics=training_result['train_metrics'],
            validation_metrics=training_result['validation_metrics'],
            model_path=model_path
        )
        
        logger.info("Model training completed", 
                   model_version=response.model_version,
                   training_time=response.training_time)
        
        return response
        
    except Exception as e:
        logger.error("Model training failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@router.post("/model/load")
async def load_model(model_path: str, detector: FraudDetector = Depends(get_fraud_detector)):
    """Load a model from file."""
    try:
        detector.load_model(model_path)
        
        logger.info("Model loaded", model_path=model_path)
        
        return {"status": "success", "message": f"Model loaded from {model_path}"}
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e), model_path=model_path)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    try:
        uptime = (datetime.now() - start_time).total_seconds()
        model_info = fraud_detector.get_model_info()
        
        metrics = {
            "uptime_seconds": uptime,
            "model_loaded": model_info.get('status') != 'no_model_loaded',
            "model_version": model_info.get('version'),
            "feature_count": model_info.get('feature_count', 0),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception", error=str(exc))
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    ) 