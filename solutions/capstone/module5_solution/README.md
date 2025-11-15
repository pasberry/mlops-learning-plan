# Capstone Module 5: Ranking Service & Deployment

## Overview

Deploys the two-tower model for production serving with real-time and batch ranking capabilities.

## Service Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────>│ Model Server │────>│    Redis     │
│   Gateway    │     │  (PyTorch)   │     │ (Embeddings) │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐
│  Monitoring  │
│   (Metrics)  │
└──────────────┘
```

## Ranking Service

### FastAPI Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI(title="Feed Ranking Service")

# Global model and cache
model = None
item_embeddings = None
feature_cache = None

class RankingRequest(BaseModel):
    user_id: int
    num_items: int = 50
    context: dict = {}

class RankingResponse(BaseModel):
    user_id: int
    items: List[Dict[str, Any]]
    latency_ms: float
    model_version: str

@app.on_event("startup")
async def load_model():
    """Load model and embeddings on startup."""
    global model, item_embeddings, feature_cache

    # Load model
    model = load_two_tower_model('models/production/model.pt')
    model.eval()

    # Load pre-computed item embeddings
    item_embeddings = load_item_embeddings_from_redis()

    # Initialize feature cache
    feature_cache = FeatureCache()

    logger.info(f"Model loaded: {model.version}")
    logger.info(f"Item embeddings: {len(item_embeddings)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "item_count": len(item_embeddings) if item_embeddings else 0
    }

@app.post("/rank", response_model=RankingResponse)
async def rank_feed(request: RankingRequest):
    """
    Generate personalized feed ranking.
    """
    start_time = time.time()

    try:
        # Get user features
        user_features = feature_cache.get_user_features(request.user_id)
        if user_features is None:
            raise HTTPException(404, "User not found")

        # Get user embedding
        with torch.no_grad():
            user_emb = model.get_user_embedding(
                torch.FloatTensor(user_features).unsqueeze(0)
            ).numpy().squeeze()

        # Retrieve top candidates using ANN
        candidate_items = retrieve_candidates_ann(
            user_emb,
            item_embeddings,
            k=request.num_items * 2  # 2x for re-ranking
        )

        # Re-rank with full model
        ranked_items = rerank_with_context(
            model,
            user_features,
            candidate_items,
            request.context,
            k=request.num_items
        )

        # Apply business rules
        final_items = apply_business_rules(
            ranked_items,
            user_id=request.user_id,
            context=request.context
        )

        latency_ms = (time.time() - start_time) * 1000

        return RankingResponse(
            user_id=request.user_id,
            items=final_items,
            latency_ms=latency_ms,
            model_version=model.version
        )

    except Exception as e:
        logger.error(f"Ranking failed: {str(e)}")
        raise HTTPException(500, str(e))
```

### Candidate Retrieval

```python
def retrieve_candidates_ann(user_emb, item_embeddings, k=100):
    """
    Fast approximate nearest neighbor search.
    """
    # Option 1: FAISS (for large scale)
    import faiss

    # Build index (done once at startup)
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product
    index.add(item_embeddings_matrix)

    # Search
    scores, indices = index.search(user_emb.reshape(1, -1), k)

    return indices[0]

    # Option 2: Simple numpy (for smaller scale)
    # scores = user_emb @ item_embeddings.T
    # top_k_indices = np.argsort(scores)[-k:][::-1]
    # return top_k_indices
```

### Re-Ranking

```python
def rerank_with_context(model, user_features, candidate_items, context, k=50):
    """
    Re-rank candidates with contextual features.
    """
    # Add context to features
    user_features_context = add_context_features(user_features, context)

    # Get item features
    item_features = [get_item_features(item_id) for item_id in candidate_items]

    # Score with full model
    with torch.no_grad():
        scores = model(
            torch.FloatTensor([user_features_context] * len(item_features)),
            torch.FloatTensor(item_features)
        ).numpy().squeeze()

    # Sort and return top k
    ranked_indices = np.argsort(scores)[::-1][:k]
    ranked_items = [
        {
            'item_id': candidate_items[i],
            'score': float(scores[i]),
            'rank': rank + 1
        }
        for rank, i in enumerate(ranked_indices)
    ]

    return ranked_items
```

### Business Rules

```python
def apply_business_rules(items, user_id, context):
    """
    Apply business logic to rankings.
    """
    # 1. Diversity: Max 3 items per category
    items = enforce_category_diversity(items, max_per_category=3)

    # 2. Freshness: Boost recent content
    items = apply_freshness_boost(items, boost_hours=24, boost_factor=1.2)

    # 3. Creator diversity: Max 2 items per creator
    items = enforce_creator_diversity(items, max_per_creator=2)

    # 4. Quality filter: Remove low-quality items
    items = filter_low_quality(items, min_quality_score=0.5)

    # 5. Deduplication: Remove recently shown items
    items = remove_recently_shown(items, user_id, hours=24)

    # 6. Promoted content: Insert sponsored items
    items = insert_promoted_content(items, positions=[3, 10, 20])

    return items
```

## Batch Ranking

### Daily Pre-Computation

```python
@daily_job
def batch_compute_rankings():
    """
    Pre-compute rankings for all users.
    Runs daily at 5 AM.
    """
    # Load model
    model = load_model()

    # Get all users
    all_users = get_active_users(days=30)

    # Batch processing
    batch_size = 1000

    for i in range(0, len(all_users), batch_size):
        batch_users = all_users[i:i+batch_size]

        # Get user features
        user_features = get_batch_user_features(batch_users)

        # Get user embeddings
        with torch.no_grad():
            user_embs = model.get_user_embedding(
                torch.FloatTensor(user_features)
            ).numpy()

        # Compute rankings for each user
        for j, user_id in enumerate(batch_users):
            # Retrieve candidates
            candidates = retrieve_candidates_ann(user_embs[j], item_embeddings)

            # Re-rank
            rankings = rerank_candidates(model, user_id, candidates)

            # Cache rankings
            cache_user_rankings(user_id, rankings, ttl=86400)  # 24h

        logger.info(f"Processed {i+batch_size}/{len(all_users)} users")
```

## Caching Strategy

### Multi-Tier Caching

```python
class RankingCache:
    """
    L1: User rankings (TTL: 12h)
    L2: User embeddings (TTL: 1h)
    L3: Item embeddings (TTL: 24h)
    L4: Features (TTL: 30m)
    """

    def __init__(self):
        self.redis = RedisClient()

    def get_user_rankings(self, user_id):
        """Get cached rankings for user."""
        key = f"rankings:user:{user_id}"
        rankings = self.redis.get(key)

        if rankings:
            logger.info(f"Cache hit: rankings for user {user_id}")
            return json.loads(rankings)

        return None

    def set_user_rankings(self, user_id, rankings, ttl=43200):
        """Cache user rankings (12h TTL)."""
        key = f"rankings:user:{user_id}"
        self.redis.setex(key, ttl, json.dumps(rankings))

    def get_user_embedding(self, user_id):
        """Get cached user embedding."""
        key = f"embedding:user:{user_id}"
        emb_bytes = self.redis.get(key)

        if emb_bytes:
            return np.frombuffer(emb_bytes, dtype=np.float32)

        return None

    def set_user_embedding(self, user_id, embedding, ttl=3600):
        """Cache user embedding (1h TTL)."""
        key = f"embedding:user:{user_id}"
        self.redis.setex(key, ttl, embedding.tobytes())
```

## Performance Optimization

### Model Optimization

```python
# 1. Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 2. TorchScript
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model_scripted.pt')

# 3. ONNX (for cross-platform)
torch.onnx.export(
    model,
    (sample_user_features, sample_item_features),
    'model.onnx',
    input_names=['user_features', 'item_features'],
    output_names=['score']
)
```

### Load Balancing

```python
# Deploy multiple replicas
# docker-compose.yml
services:
  ranking_service_1:
    image: ranking-service:latest
    ports:
      - "8000:8000"

  ranking_service_2:
    image: ranking-service:latest
    ports:
      - "8001:8000"

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    depends_on:
      - ranking_service_1
      - ranking_service_2
```

## Monitoring

### Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
ranking_requests = Counter('ranking_requests_total', 'Total ranking requests')
ranking_latency = Histogram('ranking_latency_seconds', 'Ranking latency')
ranking_errors = Counter('ranking_errors_total', 'Ranking errors')

# Model metrics
predictions_count = Counter('predictions_total', 'Total predictions')
cache_hits = Counter('cache_hits_total', 'Cache hits', ['cache_type'])

# Business metrics
items_served = Counter('items_served_total', 'Items served')
diversity_score = Gauge('feed_diversity_score', 'Feed diversity')

@app.post("/rank")
async def rank_feed(request):
    ranking_requests.inc()

    with ranking_latency.time():
        result = compute_rankings(request)

    return result
```

### Logging

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "ranking_request",
    user_id=user_id,
    num_items=num_items,
    latency_ms=latency,
    cache_hit=cache_hit,
    model_version=model_version
)
```

## Integration Points

```
Module 4 (Training) → Trained model
Module 6 (Monitoring) → Performance metrics
Module 7 (System) → Production deployment
```

## Learning Outcomes

✅ Production API design
✅ Efficient serving strategies
✅ Caching architectures
✅ Performance optimization
✅ Monitoring and observability
