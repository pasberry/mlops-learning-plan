"""
Pydantic schemas for model serving API.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""

    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary with feature names as keys",
        example={
            "age": 35,
            "income": 75000,
            "credit_score": 720,
            "num_purchases": 5
        }
    )

    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use for prediction"
    )

    @validator('features')
    def validate_features(cls, v):
        """Ensure features dict is not empty."""
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""

    instances: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries",
        min_items=1,
        max_items=1000
    )

    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use for prediction"
    )

    @validator('instances')
    def validate_instances(cls, v):
        """Ensure all instances have features."""
        if not v:
            raise ValueError("Instances list cannot be empty")
        for idx, instance in enumerate(v):
            if not instance:
                raise ValueError(f"Instance {idx} is empty")
        return v


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: float = Field(
        ...,
        description="Model prediction (probability or score)"
    )

    prediction_class: Optional[int] = Field(
        None,
        description="Predicted class (for classification)"
    )

    model_version: str = Field(
        ...,
        description="Model version used for prediction"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )

    latency_ms: Optional[float] = Field(
        None,
        description="Inference latency in milliseconds"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""

    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )

    total_count: int = Field(
        ...,
        description="Total number of predictions"
    )

    model_version: str = Field(
        ...,
        description="Model version used for predictions"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Batch prediction timestamp"
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )

    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded"
    )

    model_version: Optional[str] = Field(
        None,
        description="Loaded model version"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )


class ModelInfoResponse(BaseModel):
    """Response schema for model info endpoint."""

    model_name: str = Field(
        ...,
        description="Model name"
    )

    model_version: str = Field(
        ...,
        description="Model version"
    )

    model_type: str = Field(
        ...,
        description="Model type/architecture"
    )

    input_features: List[str] = Field(
        ...,
        description="Expected input features"
    )

    created_at: Optional[datetime] = Field(
        None,
        description="Model creation timestamp"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional model metadata"
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(
        ...,
        description="Error message"
    )

    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
