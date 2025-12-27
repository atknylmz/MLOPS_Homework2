"""
FastAPI REST API for High-Cardinality Prediction Service

This module provides the HTTP endpoints for the prediction service.
It integrates feature engineering and model prediction into a cohesive API.
"""

import os
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.feature_engineering import FeatureEngineer, FeatureHasher
from src.model import PredictionModel


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    user_id: str = Field(..., description="User identifier (high-cardinality)")
    product_id: str = Field(..., description="Product identifier (high-cardinality)")
    category: str = Field(..., description="Product category")
    price: float = Field(..., ge=0, description="Product price")
    quantity: int = Field(..., ge=1, description="Quantity")
    user_age: Optional[float] = Field(None, ge=0, le=150, description="User age")
    session_duration: Optional[float] = Field(None, ge=0, description="Session duration in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_12345",
                    "product_id": "prod_67890",
                    "category": "electronics",
                    "price": 299.99,
                    "quantity": 1,
                    "user_age": 35,
                    "session_duration": 120.5
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: int = Field(..., description="Predicted class (0 or 1)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_version: str = Field(..., description="Model version used")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction endpoint."""

    requests: List[PredictionRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint."""

    predictions: List[PredictionResponse]
    batch_size: int


# ============================================================================
# Application Setup
# ============================================================================

# Create FastAPI application
app = FastAPI(
    title="High-Cardinality Prediction Service",
    description="ML prediction service with hash-based feature engineering for high-cardinality categorical features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize components
CATEGORICAL_FEATURES = ["user_id", "product_id", "category"]
NUMERICAL_FEATURES = ["price", "quantity", "user_age", "session_duration"]
N_HASH_BUCKETS = 1024

feature_engineer = FeatureEngineer(
    categorical_features=CATEGORICAL_FEATURES,
    numerical_features=NUMERICAL_FEATURES,
    n_hash_buckets=N_HASH_BUCKETS
)

# Load model
model_path = os.environ.get("MODEL_PATH", None)
prediction_model = PredictionModel(model_path)
prediction_model.load()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "High-Cardinality Prediction Service",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for deployment verification.

    This endpoint is used by the smoke test to verify
    the service is up and responding correctly.

    Returns:
        HealthResponse: Service health status
    """
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_model.is_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction based on input features.

    This endpoint:
    1. Receives raw features from the request
    2. Applies feature engineering (hashing for categorical features)
    3. Runs the ML model
    4. Returns the prediction with confidence

    Args:
        request: PredictionRequest with input features

    Returns:
        PredictionResponse with prediction and confidence

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not prediction_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    try:
        # Convert request to dictionary for feature engineering
        input_data = {
            "user_id": request.user_id,
            "product_id": request.product_id,
            "category": request.category,
            "price": request.price,
            "quantity": request.quantity,
            "user_age": request.user_age or 0,
            "session_duration": request.session_duration or 0
        }

        # Apply feature engineering
        features = feature_engineer.transform(input_data)

        # Make prediction
        result = prediction_model.predict(features)

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_version=result["model_version"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for a batch of requests.

    Args:
        request: BatchPredictionRequest with list of prediction requests

    Returns:
        BatchPredictionResponse with list of predictions
    """
    if not prediction_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    predictions = []
    for pred_request in request.requests:
        input_data = {
            "user_id": pred_request.user_id,
            "product_id": pred_request.product_id,
            "category": pred_request.category,
            "price": pred_request.price,
            "quantity": pred_request.quantity,
            "user_age": pred_request.user_age or 0,
            "session_duration": pred_request.session_duration or 0
        }

        features = feature_engineer.transform(input_data)
        result = prediction_model.predict(features)

        predictions.append(PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_version=result["model_version"]
        ))

    return BatchPredictionResponse(
        predictions=predictions,
        batch_size=len(predictions)
    )


@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """Get information about the loaded model."""
    return prediction_model.get_model_info()


@app.get("/features/hash")
async def hash_feature(value: str, n_buckets: int = 1024):
    """
    Utility endpoint to hash a single value.

    Useful for debugging and understanding feature hashing.

    Args:
        value: String value to hash
        n_buckets: Number of hash buckets

    Returns:
        Dictionary with the hash bucket index
    """
    hasher = FeatureHasher(n_buckets=n_buckets)
    bucket = hasher.hash_value(value)
    return {"value": value, "bucket": bucket, "n_buckets": n_buckets}


# ============================================================================
# Application Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    print("Starting High-Cardinality Prediction Service...")
    print(f"Model loaded: {prediction_model.is_loaded}")
    print(f"Feature dimension: {feature_engineer.get_feature_dimension()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    print("Shutting down High-Cardinality Prediction Service...")


# ============================================================================
# Main entry point for uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
