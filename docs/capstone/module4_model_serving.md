# Module 4: Model Serving with FastAPI

**Estimated Time**: 2 days
**Difficulty**: Medium

## Learning Objectives

By the end of this module, you will be able to:
- ✅ Build production-grade REST APIs with FastAPI
- ✅ Load and serve PyTorch models efficiently
- ✅ Implement request validation with Pydantic
- ✅ Log predictions for monitoring and retraining
- ✅ Add health checks and metrics endpoints
- ✅ Handle errors and edge cases gracefully
- ✅ Optimize for low latency (<100ms p99)

## Overview

This module builds a **production-ready model serving API** that:
1. Loads the trained model from MLflow
2. Serves predictions via REST endpoint
3. Validates requests and responses
4. Logs all predictions for monitoring
5. Provides health checks and metrics
6. Handles errors gracefully

**Key Principle**: The serving layer is where your model meets users. Reliability, latency, and observability are critical.

## Background

### Model Serving Patterns

**Online Serving** (Real-time):
- User sends request → Get prediction immediately
- Latency critical (<100ms)
- Used for: Feed ranking, recommendations, ads
- **We'll implement this**

**Batch Serving** (Offline):
- Process large batches of predictions
- Latency not critical (can take minutes/hours)
- Used for: Email campaigns, daily digests

**Edge Serving**:
- Model runs on device (phone, browser)
- Zero latency, privacy benefits
- Limited model complexity

### FastAPI for ML Serving

**Why FastAPI?**
- ✅ Fast (comparable to Node.js/Go)
- ✅ Automatic API documentation (Swagger/OpenAPI)
- ✅ Request validation with Pydantic
- ✅ Async support (important for high throughput)
- ✅ Easy to test and deploy

**Alternatives**:
- **TorchServe**: PyTorch's official serving (more features, more complex)
- **TensorFlow Serving**: TensorFlow's official (highly optimized)
- **BentoML**: ML serving framework (batteries included)
- **Custom Flask/Django**: Simpler but slower

## Step 1: Design API Schemas

### Request/Response Models

**File**: `src/serving/schemas.py`

```python
"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for /predict endpoint."""

    user_id: int = Field(..., ge=0, description="User ID")
    item_ids: List[int] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of item IDs to rank (max 100)"
    )
    context: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Optional context (device, timestamp, etc.)"
    )

    @validator('item_ids')
    def validate_item_ids(cls, v):
        """Ensure all item IDs are non-negative."""
        if any(item_id < 0 for item_id in v):
            raise ValueError("All item IDs must be non-negative")
        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "item_ids": [1001, 1002, 1003, 1004, 1005],
                "context": {
                    "device": "mobile",
                    "timestamp": "2024-11-15T10:30:00Z"
                }
            }
        }


class ItemPrediction(BaseModel):
    """Single item prediction."""

    item_id: int
    score: float = Field(..., ge=0.0, le=1.0, description="Prediction score [0, 1]")
    rank: int = Field(..., ge=1, description="Rank position (1 = highest)")


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""

    user_id: int
    predictions: List[ItemPrediction]
    model_version: str
    latency_ms: float
    timestamp: str

    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "predictions": [
                    {"item_id": 1002, "score": 0.87, "rank": 1},
                    {"item_id": 1005, "score": 0.72, "rank": 2},
                    {"item_id": 1001, "score": 0.65, "rank": 3},
                    {"item_id": 1003, "score": 0.43, "rank": 4},
                    {"item_id": 1004, "score": 0.21, "rank": 5},
                ],
                "model_version": "v1.0.0",
                "latency_ms": 45.2,
                "timestamp": "2024-11-15T10:30:01Z"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""

    status: str = Field(..., description="'healthy' or 'unhealthy'")
    model_version: str
    model_loaded_at: str
    uptime_seconds: float
    last_prediction_at: Optional[str] = None


class MetricsResponse(BaseModel):
    """Response schema for /metrics endpoint."""

    requests_total: int
    requests_per_second: float
    latency_p50_ms: float
    latency_p99_ms: float
    error_rate: float
    predictions_total: int
```

## Step 2: Implement Model Loader

**File**: `src/serving/model_loader.py`

```python
"""
Model loading and feature preparation for serving.
"""

import torch
import mlflow.pytorch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import pickle


class ModelLoader:
    """Load and manage model for serving."""

    def __init__(
        self,
        model_uri: str = None,
        config_path: str = "config/model_config.yaml"
    ):
        """
        Args:
            model_uri: MLflow model URI (e.g., 'models:/feed_ranker/production')
                      If None, loads from local path
            config_path: Path to model config
        """
        self.model_uri = model_uri
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load model
        self.model = self._load_model()
        self.model.eval()  # Set to evaluation mode

        print(f"Model loaded on {self.device}")

    def _load_model(self) -> torch.nn.Module:
        """Load model from MLflow or local path."""
        if self.model_uri:
            # Load from MLflow
            print(f"Loading model from MLflow: {self.model_uri}")
            model = mlflow.pytorch.load_model(self.model_uri, map_location=self.device)
        else:
            # Load from local checkpoint
            print("Loading model from local checkpoint")
            from src.models.ranker import create_model

            # Infer feature dimensions from saved features
            sample_df = pd.read_parquet("data/features/train_features.parquet", nrows=1)
            feature_cols = [
                col for col in sample_df.columns
                if col not in ['click', 'timestamp', 'user_id', 'item_id']
            ]

            if self.config['model']['type'] == 'two_tower':
                user_feature_cols = [col for col in feature_cols if col.startswith('user_')]
                item_feature_cols = [col for col in feature_cols if col.startswith('item_')]
                feature_dims = {
                    'user': len(user_feature_cols),
                    'item': len(item_feature_cols)
                }
            else:
                feature_dims = {'total': len(feature_cols)}

            model = create_model(self.config, feature_dims)
            model.load_state_dict(torch.load("models/training/best_model.pt", map_location=self.device))

        return model

    def predict(
        self,
        user_id: int,
        item_ids: List[int],
        context: Dict = None
    ) -> List[Tuple[int, float]]:
        """
        Generate predictions for user-item pairs.

        Args:
            user_id: User ID
            item_ids: List of item IDs
            context: Optional context dict

        Returns:
            List of (item_id, score) tuples, sorted by score descending
        """
        # Load features
        features = self._prepare_features(user_id, item_ids, context)

        # Run inference
        with torch.no_grad():
            if self.config['model']['type'] == 'two_tower':
                user_features = features['user_features'].to(self.device)
                item_features = features['item_features'].to(self.device)
                scores = self.model(user_features, item_features)
            else:
                features_tensor = features['features'].to(self.device)
                scores = self.model(features_tensor)

        scores = scores.cpu().numpy()

        # Create (item_id, score) pairs
        predictions = list(zip(item_ids, scores))

        # Sort by score descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def _prepare_features(
        self,
        user_id: int,
        item_ids: List[int],
        context: Dict = None
    ) -> Dict:
        """
        Prepare features for inference.

        In production, this would query a feature store.
        For this project, we'll use mock features or cached aggregates.
        """
        # SIMPLIFIED: Use dummy features
        # In production, you'd:
        # 1. Query feature store for user features
        # 2. Query feature store for item features
        # 3. Compute real-time context features

        n_items = len(item_ids)

        if self.config['model']['type'] == 'two_tower':
            # User features (same for all items)
            user_features = torch.zeros(n_items, 4)  # 4 user features
            user_features[:, 0] = 0.15  # user_historical_ctr
            user_features[:, 1] = 30.0  # user_avg_dwell_time
            user_features[:, 2] = 100   # user_interaction_count
            user_features[:, 3] = 90    # user_days_active

            # Item features (different for each item)
            item_features = torch.zeros(n_items, 4)  # 4 item features
            for i in range(n_items):
                item_features[i, 0] = np.random.uniform(0.1, 0.3)  # item_ctr
                item_features[i, 1] = np.random.uniform(20, 60)    # item_avg_dwell_time
                item_features[i, 2] = np.random.uniform(0, 5)      # item_popularity (log scale)
                item_features[i, 3] = np.random.uniform(0, 100)    # item_age_days

            return {
                'user_features': user_features,
                'item_features': item_features
            }
        else:
            # Deep MLP: concatenate all features
            # Total features: 4 user + 4 item + 7 context = 15
            features = torch.zeros(n_items, 15)

            # User features (repeated for each item)
            features[:, 0] = 0.15
            features[:, 1] = 30.0
            features[:, 2] = 100
            features[:, 3] = 90

            # Item features (different per item)
            for i in range(n_items):
                features[i, 4] = np.random.uniform(0.1, 0.3)
                features[i, 5] = np.random.uniform(20, 60)
                features[i, 6] = np.random.uniform(0, 5)
                features[i, 7] = np.random.uniform(0, 100)

            # Context features
            features[:, 8] = 14  # hour_of_day
            features[:, 9] = 3   # day_of_week
            features[:, 10] = 1  # device_mobile
            features[:, 11] = 0  # device_desktop
            features[:, 12] = 0  # device_tablet
            features[:, 13] = 0  # user_item_previous_interactions

            return {'features': features}


# NOTE: In production, replace _prepare_features with real feature store queries:
#
# def _prepare_features_production(self, user_id, item_ids, context):
#     from feast import FeatureStore
#
#     store = FeatureStore(repo_path=".")
#
#     # Get user features
#     user_features = store.get_online_features(
#         features=["user_stats:historical_ctr", "user_stats:avg_dwell_time", ...],
#         entity_rows=[{"user_id": user_id}]
#     ).to_df()
#
#     # Get item features
#     item_features = store.get_online_features(
#         features=["item_stats:ctr", "item_stats:popularity", ...],
#         entity_rows=[{"item_id": item_id} for item_id in item_ids]
#     ).to_df()
#
#     # Combine and return
#     ...
```

## Step 3: Build FastAPI Application

**File**: `src/serving/app.py`

```python
"""
FastAPI application for model serving.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import time
from datetime import datetime
import uvicorn
import numpy as np
from pathlib import Path
import json
from collections import deque

from src.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    ItemPrediction,
    HealthResponse,
    MetricsResponse
)
from src.serving.model_loader import ModelLoader


# Initialize FastAPI app
app = FastAPI(
    title="Feed Ranking API",
    description="Production ML API for feed ranking predictions",
    version="1.0.0"
)

# Global state
model_loader = None
model_version = "v1.0.0"
model_loaded_at = None
app_start_time = None

# Metrics tracking
request_count = 0
prediction_count = 0
latencies = deque(maxlen=1000)  # Keep last 1000 latencies
error_count = 0
last_prediction_time = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model_loader, model_loaded_at, app_start_time

    print("Starting Feed Ranking API...")
    print("Loading model...")

    model_loader = ModelLoader()
    model_loaded_at = datetime.utcnow().isoformat()
    app_start_time = time.time()

    print("✅ Model loaded successfully!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Feed Ranking API",
        "version": model_version,
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate ranking predictions for user-item pairs.

    Returns items ranked by predicted engagement probability.
    """
    global request_count, prediction_count, last_prediction_time

    start_time = time.time()

    try:
        request_count += 1

        # Get predictions
        predictions = model_loader.predict(
            user_id=request.user_id,
            item_ids=request.item_ids,
            context=request.context
        )

        # Format response
        item_predictions = []
        for rank, (item_id, score) in enumerate(predictions, start=1):
            item_predictions.append(
                ItemPrediction(
                    item_id=item_id,
                    score=float(score),
                    rank=rank
                )
            )

        prediction_count += len(predictions)
        last_prediction_time = datetime.utcnow().isoformat()

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)

        # Log prediction (for monitoring and retraining)
        log_prediction(request, item_predictions, latency_ms)

        response = PredictionResponse(
            user_id=request.user_id,
            predictions=item_predictions,
            model_version=model_version,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat()
        )

        return response

    except Exception as e:
        global error_count
        error_count += 1

        # Log error
        print(f"Error during prediction: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    uptime = time.time() - app_start_time

    return HealthResponse(
        status="healthy",
        model_version=model_version,
        model_loaded_at=model_loaded_at,
        uptime_seconds=uptime,
        last_prediction_at=last_prediction_time
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Metrics endpoint for monitoring."""
    if not latencies:
        p50 = p99 = 0.0
    else:
        p50 = float(np.percentile(latencies, 50))
        p99 = float(np.percentile(latencies, 99))

    uptime = time.time() - app_start_time
    qps = request_count / uptime if uptime > 0 else 0.0
    error_rate = error_count / request_count if request_count > 0 else 0.0

    return MetricsResponse(
        requests_total=request_count,
        requests_per_second=qps,
        latency_p50_ms=p50,
        latency_p99_ms=p99,
        error_rate=error_rate,
        predictions_total=prediction_count
    )


def log_prediction(
    request: PredictionRequest,
    predictions: list,
    latency_ms: float
):
    """
    Log prediction for monitoring and retraining.

    In production, this would write to a data lake or streaming system.
    """
    # Create log directory if it doesn't exist
    log_dir = Path("data/predictions")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log entry
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': request.user_id,
        'item_ids': request.item_ids,
        'predictions': [
            {'item_id': p.item_id, 'score': p.score, 'rank': p.rank}
            for p in predictions
        ],
        'context': request.context,
        'model_version': model_version,
        'latency_ms': latency_ms
    }

    # Append to daily log file
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = log_dir / f"predictions_{date_str}.jsonl"

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "src.serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

## Step 4: Testing

### Unit Tests

**File**: `tests/test_api.py`

```python
"""
Unit tests for API endpoints.
"""

from fastapi.testclient import TestClient
import sys
sys.path.append('/home/user/mlops-learning-plan/capstone_project')

from src.serving.app import app


client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests_total" in data
    assert "latency_p50_ms" in data


def test_predict_valid_request():
    """Test prediction with valid request."""
    request_data = {
        "user_id": 123,
        "item_ids": [1001, 1002, 1003],
        "context": {"device": "mobile"}
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == 123
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["rank"] == 1


def test_predict_invalid_user_id():
    """Test prediction with invalid user ID."""
    request_data = {
        "user_id": -1,  # Invalid
        "item_ids": [1001, 1002],
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error


def test_predict_too_many_items():
    """Test prediction with too many items."""
    request_data = {
        "user_id": 123,
        "item_ids": list(range(200)),  # Max is 100
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 422


def test_predict_latency():
    """Test that latency is reasonable."""
    request_data = {
        "user_id": 123,
        "item_ids": list(range(50)),
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["latency_ms"] < 200  # Should be under 200ms
```

### Manual Testing

```bash
# Start server
cd capstone_project
python src/serving/app.py

# In another terminal, test endpoints

# 1. Health check
curl http://localhost:8000/health

# 2. Metrics
curl http://localhost:8000/metrics

# 3. Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "item_ids": [1001, 1002, 1003, 1004, 1005],
    "context": {"device": "mobile"}
  }'

# 4. Interactive docs
# Open http://localhost:8000/docs in browser
```

### Load Testing

```bash
# Install locust
pip install locust

# Create locustfile.py
cat > locustfile.py << 'EOF'
from locust import HttpUser, task, between

class FeedRankingUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post("/predict", json={
            "user_id": 123,
            "item_ids": [1001, 1002, 1003, 1004, 1005]
        })

    @task(0.1)
    def health(self):
        self.client.get("/health")
EOF

# Run load test
locust --host=http://localhost:8000

# Open http://localhost:8089 and start test
# Try 10 users, 2 users/second spawn rate
```

## Step 5: Deployment Script

**File**: `scripts/start_api.sh`

```bash
#!/bin/bash

# Start FastAPI server for production

cd /home/user/mlops-learning-plan/capstone_project

echo "Starting Feed Ranking API..."

# Set environment variables
export PYTHONPATH=/home/user/mlops-learning-plan/capstone_project:$PYTHONPATH
export MLFLOW_TRACKING_URI=http://localhost:5000

# Start with uvicorn (production ASGI server)
uvicorn src.serving.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --access-log

# For development (with auto-reload):
# uvicorn src.serving.app:app --reload --port 8000
```

## Review Checklist

- [ ] API starts successfully
- [ ] All endpoints work (/, /health, /metrics, /predict)
- [ ] Request validation catches invalid inputs
- [ ] Predictions are returned in <100ms p99
- [ ] Predictions are logged
- [ ] API documentation auto-generated (/docs)
- [ ] Error handling works
- [ ] Load test passes (>100 QPS)

## What to Submit

1. **Code**:
   - `src/serving/schemas.py`
   - `src/serving/model_loader.py`
   - `src/serving/app.py`
   - `tests/test_api.py`

2. **API Documentation**: Screenshot of /docs page

3. **Performance Results**:
   - Sample prediction request/response
   - Latency distribution (from /metrics)
   - Load test results

4. **Prediction Logs**: Sample from `data/predictions/`

5. **Reflection**:
   - How did you optimize for latency?
   - What would you improve for production?
   - How would you scale to 10,000 QPS?

## Next Steps

With a serving API running, proceed to [Module 5: Monitoring & Drift Detection](module5_monitoring.md)!
