"""
FastAPI application for model serving.

Production-ready model serving with health checks, monitoring, and error handling.
"""
import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelServer:
    """Model server singleton for managing model lifecycle."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.model_version: Optional[str] = None
        self.model_metadata: Dict[str, Any] = {}
        self.feature_names: list = []
        self.model_type: str = "unknown"

    def load_model(self, model_path: str):
        """Load model from disk."""
        try:
            logger.info(f"Loading model from {model_path}")

            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extract model and metadata
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Load model architecture (simplified example)
                    from torch import nn

                    # Create model architecture
                    input_dim = checkpoint.get('input_dim', 10)
                    self.model = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model = checkpoint

                # Extract metadata
                self.model_version = checkpoint.get('version', 'v1.0')
                self.model_metadata = checkpoint.get('metadata', {})
                self.feature_names = checkpoint.get('feature_names', [])
                self.model_type = checkpoint.get('model_type', 'neural_network')
            else:
                self.model = checkpoint
                self.model_version = 'v1.0'

            # Set model to evaluation mode
            self.model.eval()

            logger.info(f"Model loaded successfully. Version: {self.model_version}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features: Dict[str, Any]) -> float:
        """Make a single prediction."""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Convert features to tensor
            # In production, you'd have proper feature preprocessing
            feature_values = []
            for fname in self.feature_names:
                feature_values.append(features.get(fname, 0.0))

            # Fallback if no feature names defined
            if not feature_values:
                feature_values = list(features.values())

            # Convert to tensor
            x = torch.tensor([feature_values], dtype=torch.float32)

            # Make prediction
            with torch.no_grad():
                output = self.model(x)
                prediction = output.item()

            return prediction

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def predict_batch(self, instances: list) -> list:
        """Make batch predictions."""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Convert all instances to tensors
            batch_features = []
            for instance in instances:
                feature_values = []
                for fname in self.feature_names:
                    feature_values.append(instance.get(fname, 0.0))

                # Fallback if no feature names defined
                if not feature_values:
                    feature_values = list(instance.values())

                batch_features.append(feature_values)

            # Convert to tensor
            x = torch.tensor(batch_features, dtype=torch.float32)

            # Make predictions
            with torch.no_grad():
                outputs = self.model(x)
                predictions = outputs.squeeze().tolist()

            # Ensure predictions is a list
            if isinstance(predictions, float):
                predictions = [predictions]

            return predictions

        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise


# Global model server instance
model_server = ModelServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    # Startup
    logger.info("Starting model serving application")

    # Load model on startup
    model_path = os.getenv(
        'MODEL_PATH',
        '/home/user/mlops-learning-plan/models/production/model.pt'
    )

    if os.path.exists(model_path):
        try:
            model_server.load_model(model_path)
            logger.info("Model loaded successfully on startup")
        except Exception as e:
            logger.warning(f"Could not load model on startup: {str(e)}")
    else:
        logger.warning(f"Model not found at {model_path}")

    yield

    # Shutdown
    logger.info("Shutting down model serving application")


# Create FastAPI application
app = FastAPI(
    title="MLOps Model Serving API",
    description="Production model serving with PyTorch and FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MLOps Model Serving API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = model_server.model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_server.model_version
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if model_server.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Measure inference time
        start_time = time.time()

        # Make prediction
        prediction = model_server.predict(request.features)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Determine prediction class (threshold at 0.5 for binary classification)
        prediction_class = 1 if prediction >= 0.5 else 0

        return PredictionResponse(
            prediction=prediction,
            prediction_class=prediction_class,
            model_version=model_server.model_version or "unknown",
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    if model_server.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Make batch predictions
        predictions = model_server.predict_batch(request.instances)

        # Create response objects
        prediction_responses = []
        for pred in predictions:
            prediction_class = 1 if pred >= 0.5 else 0
            prediction_responses.append(
                PredictionResponse(
                    prediction=pred,
                    prediction_class=prediction_class,
                    model_version=model_server.model_version or "unknown"
                )
            )

        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_count=len(predictions),
            model_version=model_server.model_version or "unknown"
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if model_server.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return ModelInfoResponse(
        model_name="mlops_model",
        model_version=model_server.model_version or "unknown",
        model_type=model_server.model_type,
        input_features=model_server.feature_names,
        metadata=model_server.model_metadata
    )


@app.post("/model/load")
async def load_model(model_path: str):
    """Load a new model (admin endpoint)."""
    try:
        model_server.load_model(model_path)
        return {
            "status": "success",
            "message": f"Model loaded from {model_path}",
            "version": model_server.model_version
        }
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
