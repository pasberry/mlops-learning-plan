# Lab 4.1: Model Serving with FastAPI

**Goal**: Build a production-ready REST API for model inference

**Estimated Time**: 90-120 minutes

**Prerequisites**:
- Trained model from Phase 3
- Understanding of HTTP APIs
- FastAPI installed (`pip install fastapi uvicorn`)

---

## Learning Objectives

By the end of this lab, you will:
- ✅ Build a FastAPI application for model serving
- ✅ Load PyTorch models efficiently (once, not per request)
- ✅ Implement request/response validation with Pydantic
- ✅ Add health checks and versioning endpoints
- ✅ Test your API with example clients
- ✅ Understand production serving best practices

---

## Background: Why FastAPI for Model Serving?

### FastAPI Advantages
- **Fast**: Built on Starlette and Pydantic, async support
- **Type validation**: Automatic request/response validation
- **Documentation**: Auto-generated OpenAPI (Swagger) docs
- **Easy deployment**: ASGI-compatible, works with Gunicorn/Uvicorn
- **Production-ready**: Used by Microsoft, Uber, Netflix

### Serving Architecture

```
┌─────────────┐
│   Client    │
│ (Browser/   │
│  Service)   │
└──────┬──────┘
       │ HTTP POST /predict
       │ {"features": [...]}
       ▼
┌─────────────────┐
│  FastAPI Server │
│  ┌───────────┐  │
│  │ Validate  │  │  ← Pydantic schemas
│  │ Request   │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Load      │  │  ← Singleton model
│  │ Features  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Model     │  │  ← PyTorch inference
│  │ Inference │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Return    │  │  ← JSON response
│  │ Response  │  │
│  └───────────┘  │
└─────────────────┘
```

---

## Part 1: Create the Model Service

Create `ml/serving/model_service.py`:

```python
"""
Model Service - Handles model loading and inference
Singleton pattern: Load model once, reuse for all requests
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleClassifier(nn.Module):
    """
    Example model architecture (must match your training code)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ModelService:
    """
    Singleton service for model inference
    """
    _instance = None
    _model = None
    _model_version = None
    _feature_names = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(
        self,
        model_path: str,
        feature_names: List[str],
        device: str = "cpu"
    ):
        """
        Load model from checkpoint
        Only called once during app startup
        """
        if self._model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model from {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Extract model info
        input_dim = checkpoint.get('input_dim', len(feature_names))
        hidden_dim = checkpoint.get('hidden_dim', 64)
        output_dim = checkpoint.get('output_dim', 2)

        # Initialize model
        self._model = SimpleClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

        # Load weights
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()  # Set to evaluation mode
        self._model.to(device)

        # Store metadata
        self._model_version = checkpoint.get('version', 'unknown')
        self._feature_names = feature_names

        logger.info(f"Model loaded successfully: {self._model_version}")
        logger.info(f"Input features: {len(self._feature_names)}")

    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Make prediction for a single sample

        Args:
            features: Dictionary of feature_name -> value

        Returns:
            Dictionary with prediction, probabilities, and metadata
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate features
        if set(features.keys()) != set(self._feature_names):
            missing = set(self._feature_names) - set(features.keys())
            extra = set(features.keys()) - set(self._feature_names)
            raise ValueError(
                f"Feature mismatch. Missing: {missing}, Extra: {extra}"
            )

        # Create feature tensor in correct order
        feature_values = [features[name] for name in self._feature_names]
        x = torch.tensor([feature_values], dtype=torch.float32)

        # Inference
        with torch.no_grad():
            logits = self._model(x)
            probabilities = torch.softmax(logits, dim=1)

        # Extract results
        probs = probabilities[0].tolist()
        predicted_class = int(torch.argmax(probabilities, dim=1)[0])

        return {
            "prediction": predicted_class,
            "probabilities": {
                f"class_{i}": float(prob)
                for i, prob in enumerate(probs)
            },
            "model_version": self._model_version
        }

    def predict_batch(self, features_batch: List[Dict[str, float]]) -> List[Dict]:
        """
        Make predictions for multiple samples (more efficient)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Stack features into batch
        feature_matrix = [
            [features[name] for name in self._feature_names]
            for features in features_batch
        ]
        x = torch.tensor(feature_matrix, dtype=torch.float32)

        # Batch inference
        with torch.no_grad():
            logits = self._model(x)
            probabilities = torch.softmax(logits, dim=1)

        # Extract results
        predictions = []
        for i in range(len(features_batch)):
            probs = probabilities[i].tolist()
            predicted_class = int(torch.argmax(probabilities[i]))

            predictions.append({
                "prediction": predicted_class,
                "probabilities": {
                    f"class_{j}": float(prob)
                    for j, prob in enumerate(probs)
                },
                "model_version": self._model_version
            })

        return predictions

    def get_model_info(self) -> Dict:
        """Get model metadata"""
        return {
            "model_version": self._model_version,
            "feature_names": self._feature_names,
            "num_features": len(self._feature_names) if self._feature_names else 0,
            "model_loaded": self._model is not None
        }
```

---

## Part 2: Create API Schemas

Create `ml/serving/schemas.py`:

```python
"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import datetime


class PredictRequest(BaseModel):
    """
    Single prediction request
    """
    features: Dict[str, float] = Field(
        ...,
        description="Feature values as key-value pairs"
    )

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "age": 35.0,
                    "income": 75000.0,
                    "tenure_days": 180.0,
                    "num_purchases": 5.0
                }
            }
        }

    @validator('features')
    def validate_features(cls, v):
        """Ensure all feature values are numeric"""
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature {key} must be numeric, got {type(value)}")
        return v


class BatchPredictRequest(BaseModel):
    """
    Batch prediction request
    """
    samples: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries"
    )

    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {"age": 35.0, "income": 75000.0},
                    {"age": 42.0, "income": 95000.0}
                ]
            }
        }


class PredictResponse(BaseModel):
    """
    Single prediction response
    """
    prediction: int = Field(..., description="Predicted class")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probabilities": {
                    "class_0": 0.23,
                    "class_1": 0.77
                },
                "model_version": "v1.2.0",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchPredictResponse(BaseModel):
    """
    Batch prediction response
    """
    predictions: List[PredictResponse]
    total_samples: int
    model_version: str


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """
    Model information response
    """
    model_version: str
    feature_names: List[str]
    num_features: int
    model_loaded: bool
```

---

## Part 3: Create the FastAPI Application

Create `ml/serving/app.py`:

```python
"""
FastAPI application for model serving
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from ml.serving.model_service import ModelService
from ml.serving.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model service
model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    Load model once during startup
    """
    logger.info("Starting up model service...")

    # Load model (configure paths for your environment)
    model_path = Path("models/production/model_latest.pt")
    feature_names = [
        "age", "income", "tenure_days", "num_purchases",
        "avg_transaction_value", "days_since_last_purchase"
    ]

    try:
        model_service.load_model(
            model_path=str(model_path),
            feature_names=feature_names,
            device="cpu"
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # In production, you might want to fail startup here
        # raise

    yield

    # Shutdown
    logger.info("Shutting down model service...")


# Create FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Production model inference service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint
    Used by load balancers and monitoring systems
    """
    model_info = model_service.get_model_info()

    return HealthResponse(
        status="healthy" if model_info["model_loaded"] else "unhealthy",
        model_loaded=model_info["model_loaded"],
        model_version=model_info.get("model_version")
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Get model metadata
    """
    info = model_service.get_model_info()

    if not info["model_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Single prediction endpoint

    Example:
        POST /predict
        {
            "features": {
                "age": 35.0,
                "income": 75000.0,
                ...
            }
        }
    """
    try:
        # Make prediction
        result = model_service.predict(request.features)

        # Return response
        return PredictResponse(**result)

    except ValueError as e:
        # Invalid input
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Internal error
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    """
    Batch prediction endpoint
    More efficient for multiple samples

    Example:
        POST /predict/batch
        {
            "samples": [
                {"age": 35.0, "income": 75000.0, ...},
                {"age": 42.0, "income": 95000.0, ...}
            ]
        }
    """
    try:
        # Make batch prediction
        results = model_service.predict_batch(request.samples)

        # Convert to response objects
        predictions = [PredictResponse(**r) for r in results]

        return BatchPredictResponse(
            predictions=predictions,
            total_samples=len(predictions),
            model_version=results[0]["model_version"] if results else "unknown"
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler
    Logs errors and returns consistent error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level="info"
    )
```

---

## Part 4: Create Example Client

Create `ml/serving/client.py`:

```python
"""
Example client for testing the API
"""

import requests
from typing import Dict, List
import json


class ModelClient:
    """
    Client for interacting with the model serving API
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_model_info(self) -> Dict:
        """Get model information"""
        response = requests.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Make a single prediction

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Prediction response
        """
        payload = {"features": features}
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(self, samples: List[Dict[str, float]]) -> Dict:
        """
        Make batch predictions

        Args:
            samples: List of feature dictionaries

        Returns:
            Batch prediction response
        """
        payload = {"samples": samples}
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def main():
    """Example usage"""
    client = ModelClient()

    # Health check
    print("=== Health Check ===")
    health = client.health_check()
    print(json.dumps(health, indent=2))

    # Model info
    print("\n=== Model Info ===")
    info = client.get_model_info()
    print(json.dumps(info, indent=2))

    # Single prediction
    print("\n=== Single Prediction ===")
    features = {
        "age": 35.0,
        "income": 75000.0,
        "tenure_days": 180.0,
        "num_purchases": 5.0,
        "avg_transaction_value": 125.50,
        "days_since_last_purchase": 7.0
    }
    result = client.predict(features)
    print(json.dumps(result, indent=2))

    # Batch prediction
    print("\n=== Batch Prediction ===")
    samples = [
        {
            "age": 35.0,
            "income": 75000.0,
            "tenure_days": 180.0,
            "num_purchases": 5.0,
            "avg_transaction_value": 125.50,
            "days_since_last_purchase": 7.0
        },
        {
            "age": 42.0,
            "income": 95000.0,
            "tenure_days": 365.0,
            "num_purchases": 12.0,
            "avg_transaction_value": 200.00,
            "days_since_last_purchase": 3.0
        }
    ]
    batch_result = client.predict_batch(samples)
    print(f"Predicted {batch_result['total_samples']} samples")
    print(f"Model version: {batch_result['model_version']}")


if __name__ == "__main__":
    main()
```

---

## Part 5: Running the Service

### 1. Create Directory Structure

```bash
mkdir -p ml/serving
touch ml/__init__.py
touch ml/serving/__init__.py
```

### 2. Start the Server

```bash
# Development mode (auto-reload)
cd ml/serving
python app.py

# Or using uvicorn directly
uvicorn ml.serving.app:app --reload --port 8000

# Production mode (with workers)
gunicorn ml.serving.app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 3. Test the API

```bash
# In another terminal

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 35.0,
      "income": 75000.0,
      "tenure_days": 180.0,
      "num_purchases": 5.0,
      "avg_transaction_value": 125.50,
      "days_since_last_purchase": 7.0
    }
  }'

# Or use the Python client
python ml/serving/client.py
```

### 4. Interactive API Docs

Visit http://localhost:8000/docs for auto-generated Swagger UI

---

## Exercises

### Exercise 1: Add Preprocessing

Update `model_service.py` to include feature preprocessing:

```python
def preprocess_features(self, features: Dict[str, float]) -> Dict[str, float]:
    """
    Preprocess features (scaling, encoding, etc.)
    """
    # Example: Log transform income
    if 'income' in features:
        features['income'] = np.log1p(features['income'])

    # Example: Clip age
    if 'age' in features:
        features['age'] = np.clip(features['age'], 18, 100)

    return features
```

### Exercise 2: Add Request Logging

Log all predictions for monitoring:

```python
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # ... existing code ...

    # Log prediction
    logger.info(f"Prediction: {result['prediction']}, "
                f"Confidence: {max(result['probabilities'].values()):.2f}")

    # Could also write to database or file
    return PredictResponse(**result)
```

### Exercise 3: Add Model Warmup

Pre-run inference to warm up the model:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... load model ...

    # Warmup: Run dummy prediction
    dummy_features = {name: 0.0 for name in feature_names}
    try:
        model_service.predict(dummy_features)
        logger.info("Model warmup complete")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

    yield
```

---

## Production Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY ml/ ml/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "ml.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t model-serving .
docker run -p 8000:8000 model-serving
```

### Using Kubernetes

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
    spec:
      containers:
      - name: model-serving
        image: model-serving:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
```

---

## Performance Optimization

### 1. Batch Inference

Process multiple requests together:

```python
# Instead of processing one by one
for sample in samples:
    predict(sample)

# Batch process
predict_batch(samples)  # ~10x faster
```

### 2. Model Optimization

Convert to ONNX for faster inference:

```python
# Export to ONNX
dummy_input = torch.randn(1, input_dim)
torch.onnx.export(model, dummy_input, "model.onnx")

# Load with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```

### 3. Response Caching

Cache common predictions:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(features_tuple):
    features = dict(features_tuple)
    return model_service.predict(features)
```

---

## Key Takeaways

✅ **Load model once**: Use singleton pattern, not per-request
✅ **Validate inputs**: Use Pydantic for automatic validation
✅ **Health checks**: Essential for production monitoring
✅ **Batch when possible**: Much more efficient
✅ **Log everything**: Predictions, errors, performance
✅ **Handle errors gracefully**: Return meaningful error messages

---

## Next Steps

- ✅ Complete this lab and test your API
- ✅ Share your implementation for review
- → Move to **Lab 4.2: Batch Inference Pipeline**

---

**Congratulations! You've built a production-ready model serving API! 🚀**

**Next**: [Lab 4.2 - Batch Inference →](./lab4_2_batch_inference.md)
