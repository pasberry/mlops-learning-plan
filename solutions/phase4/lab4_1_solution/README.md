# Lab 4.1: Model Serving Solution

Complete production-ready model serving with FastAPI.

## Overview

This solution implements a RESTful API for serving PyTorch models with:
- Health checks and monitoring
- Single and batch predictions
- Model versioning support
- Error handling and validation
- Performance metrics
- CORS support

## Architecture

```
serving/
├── app.py          # FastAPI application
├── schemas.py      # Pydantic request/response schemas
test_api.py         # API test client and load testing
README.md           # This file
```

## Key Components

### 1. FastAPI Application (`serving/app.py`)

**Features:**
- Async model serving with lifespan management
- Global exception handling
- Request/response validation
- Inference latency tracking
- Model hot-loading support

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions
- `GET /model/info` - Model metadata
- `POST /model/load` - Load new model (admin)

### 2. Pydantic Schemas (`serving/schemas.py`)

Type-safe request/response models:
- `PredictionRequest` - Single prediction input
- `BatchPredictionRequest` - Batch prediction input
- `PredictionResponse` - Prediction output with metadata
- `BatchPredictionResponse` - Batch prediction output
- `HealthResponse` - Health check status
- `ModelInfoResponse` - Model information
- `ErrorResponse` - Error details

### 3. Test Client (`test_api.py`)

Comprehensive testing:
- Health check validation
- Single prediction tests
- Batch prediction tests
- Model info retrieval
- Error handling validation
- Load testing capabilities

## Setup

### 1. Install Dependencies

```bash
pip install fastapi uvicorn torch pydantic requests
```

### 2. Create a Test Model

```bash
# Create models directory
mkdir -p /home/user/mlops-learning-plan/models/production

# Create a simple test model
python3 << 'EOF'
import torch
import torch.nn as nn

# Create model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Save model with metadata
checkpoint = {
    'model_state_dict': model.state_dict(),
    'version': 'v1.0',
    'input_dim': 10,
    'model_type': 'neural_network',
    'feature_names': [
        'age', 'income', 'credit_score', 'num_purchases',
        'account_age_days', 'avg_transaction', 'num_returns',
        'is_premium', 'region', 'category_preference'
    ],
    'metadata': {
        'created_at': '2025-11-15',
        'framework': 'pytorch',
        'task': 'binary_classification'
    }
}

torch.save(
    checkpoint,
    '/home/user/mlops-learning-plan/models/production/model.pt'
)

print("✓ Test model created")
EOF
```

## Running the Server

### Method 1: Direct Python

```bash
cd /home/user/mlops-learning-plan/solutions/phase4/lab4_1_solution

# Set model path (optional, defaults to models/production/model.pt)
export MODEL_PATH=/home/user/mlops-learning-plan/models/production/model.pt

# Run server
python serving/app.py
```

### Method 2: Using Uvicorn

```bash
cd /home/user/mlops-learning-plan/solutions/phase4/lab4_1_solution

# Run with uvicorn for more control
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### Method 3: Production Deployment

```bash
# Run with multiple workers for production
uvicorn serving.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
```

Server will be available at: `http://localhost:8000`

## Testing the API

### 1. Interactive API Documentation

Open in browser:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 2. Health Check

```bash
curl http://localhost:8000/health
```

### 3. Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 35,
      "income": 75000,
      "credit_score": 720,
      "num_purchases": 5,
      "account_age_days": 365,
      "avg_transaction": 150.0,
      "num_returns": 1,
      "is_premium": 1,
      "region": 2,
      "category_preference": 3
    }
  }'
```

### 4. Batch Prediction

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "age": 25,
        "income": 45000,
        "credit_score": 650,
        "num_purchases": 2,
        "account_age_days": 180,
        "avg_transaction": 80.0,
        "num_returns": 0,
        "is_premium": 0,
        "region": 1,
        "category_preference": 1
      },
      {
        "age": 45,
        "income": 95000,
        "credit_score": 780,
        "num_purchases": 15,
        "account_age_days": 730,
        "avg_transaction": 250.0,
        "num_returns": 2,
        "is_premium": 1,
        "region": 3,
        "category_preference": 2
      }
    ]
  }'
```

### 5. Model Info

```bash
curl http://localhost:8000/model/info
```

### 6. Run Test Suite

```bash
cd /home/user/mlops-learning-plan/solutions/phase4/lab4_1_solution

# Run all tests
python test_api.py

# Run with load test
python test_api.py --load-test --num-requests 100

# Test against custom URL
python test_api.py --url http://localhost:8000
```

## Expected Output

### Health Check Response
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1.0",
  "timestamp": "2025-11-15T10:30:00.123456"
}
```

### Prediction Response
```json
{
  "prediction": 0.7234,
  "prediction_class": 1,
  "model_version": "v1.0",
  "timestamp": "2025-11-15T10:30:00.123456",
  "latency_ms": 2.34
}
```

### Test Suite Output
```
============================================================
MLOps Model Serving API Tests
============================================================
Target: http://localhost:8000

=== Testing Health Check ===
Status: healthy
Model Loaded: True
Model Version: v1.0
✓ Health check passed

=== Testing Single Prediction ===
Prediction: 0.7234
Prediction Class: 1
Model Version: v1.0
API Latency: 15.23ms
✓ Single prediction passed

...

============================================================
Test Summary
============================================================
Health Check................................ ✓ PASS
Single Prediction........................... ✓ PASS
Batch Prediction............................ ✓ PASS
Model Info.................................. ✓ PASS
Error Handling.............................. ✓ PASS

Total: 5/5 tests passed

🎉 All tests passed!
```

## Performance Considerations

### Batch Processing
- Use `/batch_predict` for multiple predictions
- Reduces overhead compared to individual requests
- Typical batch size: 10-1000 instances

### Model Loading
- Model loaded once at startup (lifespan event)
- Hot-loading supported via `/model/load` endpoint
- Models run in eval mode for inference

### Concurrency
- FastAPI is async by default
- Use multiple workers for CPU-bound workloads
- Consider GPU batching for larger models

## Production Enhancements

### 1. Add Authentication

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    # ... prediction logic
```

### 2. Add Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, ...):
    # ... prediction logic
```

### 3. Add Monitoring

```python
from prometheus_client import Counter, Histogram, make_asgi_app

PREDICTION_COUNT = Counter('predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(...):
    with PREDICTION_LATENCY.time():
        result = model_server.predict(...)
    PREDICTION_COUNT.inc()
    return result

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### 4. Add Model Caching

```python
from functools import lru_cache

class ModelServer:
    @lru_cache(maxsize=100)
    def predict_cached(self, features_hash: str, features: dict):
        return self.predict(features)
```

## Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/

ENV MODEL_PATH=/models/model.pt

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t mlops-serving .

# Run
docker run -p 8000:8000 \
  -v /path/to/models:/models \
  -e MODEL_PATH=/models/model.pt \
  mlops-serving
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlops-serving
  template:
    metadata:
      labels:
        app: mlops-serving
    spec:
      containers:
      - name: serving
        image: mlops-serving:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: /models/model.pt
        volumeMounts:
        - name: models
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: Service
metadata:
  name: mlops-serving
spec:
  selector:
    app: mlops-serving
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Troubleshooting

### Model Not Loading
- Check MODEL_PATH environment variable
- Verify model file exists and is readable
- Check model checkpoint format

### Slow Predictions
- Profile model inference time
- Consider model optimization (quantization, pruning)
- Use batch predictions when possible
- Add GPU support if available

### Memory Issues
- Monitor model memory usage
- Implement request size limits
- Use model quantization
- Scale horizontally with more replicas

## Next Steps

1. **Lab 4.2**: Implement batch inference pipeline
2. **Lab 4.3**: Add monitoring and drift detection
3. **Lab 4.4**: Implement automated retraining
4. **Integration**: Connect serving to full MLOps pipeline

## Learning Outcomes

After completing this lab, you should understand:
- ✅ Building production REST APIs with FastAPI
- ✅ Serving PyTorch models at scale
- ✅ Request/response validation with Pydantic
- ✅ API testing and load testing
- ✅ Model lifecycle management
- ✅ Performance optimization strategies
- ✅ Production deployment patterns
