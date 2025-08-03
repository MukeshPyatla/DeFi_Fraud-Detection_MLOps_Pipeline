"""
Main FastAPI application for DeFi fraud detection API.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import structlog

from .endpoints import router
from ..utils.logger import setup_logger
from ..utils.config import config


# Setup logging
logger = setup_logger("api")

# Create FastAPI app
app = FastAPI(
    title="DeFi Fraud Detection API",
    description="API for detecting fraud in DeFi transactions using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api.security.cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on your deployment
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Log request
    logger.info("Request started",
                method=request.method,
                url=str(request.url),
                client_host=request.client.host if request.client else None)
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info("Request completed",
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code,
                    process_time=process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error("Request failed",
                    method=request.method,
                    url=str(request.url),
                    error=str(e),
                    process_time=process_time)
        raise

# Include routers
app.include_router(router, prefix="/api/v1", tags=["fraud-detection"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "DeFi Fraud Detection API",
        "version": "1.0.0",
        "description": "Machine learning API for detecting fraud in DeFi transactions",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Health check at root level
@app.get("/health")
async def root_health():
    """Root level health check."""
    return {"status": "healthy", "service": "defi-fraud-detection-api"}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error("Unhandled exception",
                method=request.method,
                url=str(request.url),
                error=str(exc))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("API starting up",
                title=app.title,
                version=app.version,
                docs_url=app.docs_url)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("API shutting down")


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8000)
    workers = config.get('api.workers', 4)
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=True  # Enable auto-reload for development
    ) 