"""
FastAPI API for DeFi fraud detection.
"""

from .main import app
from .models import PredictionRequest, PredictionResponse, HealthResponse
from .endpoints import router

__all__ = [
    "app",
    "PredictionRequest", 
    "PredictionResponse",
    "HealthResponse",
    "router"
] 