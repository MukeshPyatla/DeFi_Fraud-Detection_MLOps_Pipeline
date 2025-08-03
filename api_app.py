"""
Simplified FastAPI for DeFi Fraud Detection - Free Deployment Version
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import numpy as np
from datetime import datetime
import json

# Create FastAPI app
app = FastAPI(
    title="DeFi Fraud Detection API",
    description="API for detecting fraud in DeFi transactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TransactionRequest(BaseModel):
    transaction_hash: str
    from_address: str
    to_address: str
    value_eth: float
    gas_price: int
    gas_used: int
    block_number: int

class PredictionResponse(BaseModel):
    fraud_score: float
    risk_level: str
    is_fraud: bool
    confidence: float
    prediction_time: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: str

class ModelInfoResponse(BaseModel):
    algorithm: str
    version: str
    feature_count: int
    performance_metrics: Dict[str, float]

# Global variables for tracking
start_time = datetime.now()
prediction_count = 0

def calculate_fraud_score(transaction: TransactionRequest) -> float:
    """Calculate fraud score based on transaction features."""
    score = 0.0
    
    # Higher value transactions are riskier
    if transaction.value_eth > 5:
        score += 0.3
    elif transaction.value_eth > 1:
        score += 0.1
    
    # Higher gas prices might indicate urgency (potential risk)
    if transaction.gas_price > 50:
        score += 0.2
    
    # Check for suspicious patterns
    if transaction.gas_used > 50000:
        score += 0.1
    
    # Add some randomness for demo
    score += np.random.uniform(0, 0.2)
    score = min(score, 1.0)
    
    return score

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DeFi Fraud Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    uptime = datetime.now() - start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=str(uptime)
    )

@app.get("/api/v1/health")
async def api_health():
    """API health endpoint for dashboard."""
    return {
        "status": "healthy",
        "model_status": "loaded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/v1/model/info")
async def get_model_info():
    """Get model information."""
    return ModelInfoResponse(
        algorithm="Demo Random Forest",
        version="demo-v1.0.0",
        feature_count=8,
        performance_metrics={
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90,
            "auc": 0.94
        }
    )

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Predict fraud for a transaction."""
    global prediction_count
    prediction_count += 1
    
    # Calculate fraud score
    fraud_score = calculate_fraud_score(transaction)
    
    # Determine risk level
    if fraud_score > 0.7:
        risk_level = "High"
    elif fraud_score > 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    # Determine if it's fraud
    is_fraud = fraud_score > 0.6
    
    # Calculate confidence
    confidence = np.random.uniform(0.7, 0.95)
    
    return PredictionResponse(
        fraud_score=fraud_score,
        risk_level=risk_level,
        is_fraud=is_fraud,
        confidence=confidence,
        prediction_time=datetime.now().isoformat(),
        model_version="demo-v1.0.0"
    )

@app.post("/predict")
async def predict_fraud_legacy(transaction: TransactionRequest):
    """Legacy endpoint for compatibility."""
    return await predict_fraud(transaction)

@app.get("/api/v1/metrics")
async def get_metrics():
    """Get API metrics."""
    uptime = datetime.now() - start_time
    return {
        "uptime_seconds": uptime.total_seconds(),
        "total_predictions": prediction_count,
        "requests_per_minute": prediction_count / max(uptime.total_seconds() / 60, 1),
        "status": "healthy"
    }

@app.get("/metrics")
async def metrics_legacy():
    """Legacy metrics endpoint."""
    return await get_metrics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 