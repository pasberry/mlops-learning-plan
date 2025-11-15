# Capstone Module 7: Feed Ranking System - Complete Integration

**The culmination of your MLOps journey** - a production-grade personalized feed ranking system.

## Overview

This capstone integrates everything you've learned into a real-world system:

### What You're Building

A **personalized content feed ranking system** similar to:
- Facebook/Instagram feed
- Twitter/X timeline
- LinkedIn feed
- TikTok For You page
- YouTube recommendations

### System Components

```
┌─────────────────────────────────────────────────────────┐
│         PERSONALIZED FEED RANKING SYSTEM                 │
└─────────────────────────────────────────────────────────┘

User Activity → Feature Extraction → Two-Tower Model
     ↓                                      ↓
Item Catalog                         Ranking Generation
     ↓                                      ↓
Real-time Serving ← Rankings ← Quality Monitoring
```

## Architecture

### 1. Two-Tower Neural Architecture

```python
class TwoTowerModel(nn.Module):
    """
    User Tower:  [user_features] → [embeddings]
    Item Tower:  [item_features] → [embeddings]
    Scoring:     dot_product(user_emb, item_emb) → [score]
    """
```

**Why Two-Tower?**
- ✅ Scalable: Compute embeddings independently
- ✅ Fast inference: Pre-compute item embeddings
- ✅ Flexible: Easy to update user/item features
- ✅ Production-ready: Used by Google, Meta, Pinterest

### 2. Complete MLOps Pipeline

```
Daily Schedule (2 AM):
1. Extract user activity features
2. Update item catalog features
3. Generate training pairs
4. Train/update two-tower model
5. Generate personalized rankings
6. Monitor quality metrics
7. Retrain if performance degrades
```

### 3. Data Pipeline

**User Features:**
- Total interactions
- Unique items viewed
- Like rate
- Average dwell time
- Platform, country, demographics

**Item Features:**
- Category, tags
- Creator metrics
- Age/freshness
- Historical engagement

**Interaction Data:**
- Views, likes, shares, comments
- Dwell time
- Timestamp
- Context (device, location)

## Quick Start

### 1. Run Demo

```bash
cd /home/user/mlops-learning-plan/solutions/capstone/module7_solution

# Make demo script executable
chmod +x scripts/demo.sh

# Run complete demo
./scripts/demo.sh
```

### 2. What the Demo Does

```
✓ Creates directory structure
✓ Generates 1,000 users
✓ Generates 5,000 items
✓ Generates 50,000 interactions
✓ Extracts user and item features
✓ Deploys DAG to Airflow
✓ Provides next steps
```

### 3. Trigger Pipeline

```bash
# Start Airflow (if not running)
export AIRFLOW_HOME=/home/user/mlops-learning-plan/airflow
airflow webserver --port 8080 &
airflow scheduler &

# Trigger feed ranking pipeline
airflow dags trigger feed_ranking_master

# Monitor progress
airflow dags list-runs -d feed_ranking_master --state running
```

## System Components

### Feature Extraction

**User Activity Analysis:**
```python
def extract_user_features(activity_df):
    """
    Aggregate user behavior metrics:
    - Total interactions (volume)
    - Unique items (diversity)
    - Like rate (preference clarity)
    - Avg dwell time (engagement depth)
    """
```

**Item Catalog Processing:**
```python
def extract_item_features(items_df):
    """
    Process content metadata:
    - Category, tags
    - Creator information
    - Freshness/age
    - Popularity metrics
    """
```

### Training Data Generation

**Positive/Negative Sampling:**
```python
# Positive: Actual user-item interactions
positive_pairs = [(user_id, item_id, 1) for interactions]

# Negative: Random non-interactions
negative_pairs = [(user_id, random_item_id, 0) for sampling]

# Balanced training set
training_data = positive_pairs + negative_pairs
```

### Two-Tower Model

**Architecture:**
```
User Tower:
  Input: [user_features]
    ↓
  Dense(128) + ReLU + Dropout
    ↓
  Dense(64) [user_embedding]

Item Tower:
  Input: [item_features]
    ↓
  Dense(128) + ReLU + Dropout
    ↓
  Dense(64) [item_embedding]

Scoring:
  score = sigmoid(dot_product(user_emb, item_emb))
```

### Ranking Generation

**Candidate Retrieval:**
```python
# 1. Get user embedding
user_emb = user_tower(user_features)

# 2. Get all item embeddings (pre-computed)
item_embs = item_tower(all_items)

# 3. Compute scores
scores = user_emb @ item_embs.T

# 4. Rank and filter
top_items = argsort(scores)[:50]
```

### Quality Monitoring

**Metrics Tracked:**
- User coverage: % users with rankings
- Item coverage: % items appearing in any feed
- Average items per feed
- Score distribution
- Diversity metrics

## Expected Output

### System Execution Report

```
================================================================================
FEED RANKING SYSTEM - EXECUTION REPORT
================================================================================
Execution Time: 2025-11-15T02:00:00.000000

Data Processing:
  Users processed: 1000
  Items processed: 5000

Ranking Generation:
  Total rankings: 50000
  Users with feeds: 1000
  Avg items/user: 50.0
  Item coverage: 87.4%
================================================================================
Feed ranking pipeline completed successfully!
================================================================================
```

### Generated Rankings

```csv
user_id,item_id,score,rank
1,4523,0.8745,1
1,2341,0.8523,2
1,1876,0.8412,3
...
2,3456,0.9123,1
2,4521,0.8934,2
...
```

## Production Deployment

### Real-Time Serving

```python
# FastAPI endpoint for real-time ranking
@app.get("/feed/{user_id}")
async def get_user_feed(user_id: int):
    """
    1. Get user embedding (cached)
    2. Retrieve candidate items
    3. Rank with two-tower model
    4. Apply business rules
    5. Return top N items
    """
```

### Batch Pre-Computation

```python
# Daily job: Pre-compute all rankings
def batch_ranking_job():
    """
    1. Load all user/item features
    2. Compute embeddings in batches
    3. Generate rankings for all users
    4. Cache in Redis/storage
    5. Serve from cache
    """
```

### Online Learning

```python
# Update model with fresh interactions
def online_update():
    """
    1. Collect recent interactions
    2. Sample training pairs
    3. Fine-tune model
    4. A/B test new model
    5. Promote if better
    """
```

## Advanced Features

### 1. Diversity Re-Ranking

```python
def diversify_rankings(ranked_items, user_history):
    """
    Apply MMR (Maximal Marginal Relevance):
    - Balance relevance vs diversity
    - Avoid filter bubble
    - Promote exploration
    """
```

### 2. Business Rules

```python
def apply_business_rules(rankings):
    """
    - Freshness boost (recent content)
    - Creator diversity (multiple sources)
    - Category limits (max 3 per category)
    - Promoted content slots
    """
```

### 3. A/B Testing

```python
def serve_with_experiment(user_id, rankings):
    """
    - Assign user to experiment
    - Serve control or treatment
    - Track engagement metrics
    - Evaluate and promote winner
    """
```

### 4. Contextual Ranking

```python
def contextual_features(user_context):
    """
    Additional context:
    - Time of day
    - Day of week
    - Device type
    - Location
    - Recent activity
    """
```

## Performance Optimization

### Embedding Caching

```python
# Cache item embeddings (change infrequently)
@lru_cache(maxsize=10000)
def get_item_embedding(item_id):
    return item_tower(item_features[item_id])

# Compute user embedding on-demand
def get_user_embedding(user_id):
    return user_tower(user_features[user_id])
```

### Approximate Nearest Neighbors

```python
# Use FAISS for fast similarity search
import faiss

# Build index
index = faiss.IndexFlatIP(embedding_dim)
index.add(item_embeddings)

# Fast retrieval
D, I = index.search(user_embedding, k=50)
```

### Model Quantization

```python
# Reduce model size for deployment
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## Monitoring & Metrics

### Business Metrics

```python
metrics = {
    'engagement_rate': clicks / impressions,
    'dwell_time': avg_time_spent,
    'retention': returning_users / total_users,
    'diversity': unique_creators_in_feed
}
```

### ML Metrics

```python
metrics = {
    'auc_roc': model_auc,
    'ndcg': normalized_dcg_score,
    'precision_at_k': precision_at_50,
    'recall_at_k': recall_at_50
}
```

### System Metrics

```python
metrics = {
    'latency_p50': 50,  # ms
    'latency_p95': 150,  # ms
    'throughput': 1000,  # requests/sec
    'cache_hit_rate': 0.95
}
```

## Integration with Previous Labs

```
Phase 1 (Foundations):
└─→ Airflow orchestration ✓

Phase 2 (Data):
└─→ Feature engineering ✓
└─→ Data validation ✓

Phase 3 (Training):
└─→ PyTorch two-tower model ✓
└─→ Training pipeline ✓

Phase 4 (Deployment):
└─→ Batch ranking (Lab 4.2) ✓
└─→ Drift monitoring (Lab 4.3) ✓
└─→ Auto-retraining (Lab 4.4) ✓
└─→ Master pipeline (Lab 4.5) ✓
```

## Real-World Applications

This system architecture powers:

**Social Media Feeds:**
- Facebook News Feed
- Instagram Explore
- Twitter Timeline
- LinkedIn Feed

**Video Platforms:**
- YouTube Homepage
- TikTok For You
- Netflix Recommendations

**E-commerce:**
- Amazon product recommendations
- Shopify personalized stores

**Content Platforms:**
- Medium recommended stories
- Spotify Discover Weekly
- Pinterest boards

## Extending the System

### Add More Towers

```python
# Three-tower: User + Item + Context
class ThreeTowerModel(nn.Module):
    def __init__(self):
        self.user_tower = ...
        self.item_tower = ...
        self.context_tower = ...  # Time, location, device

    def forward(self, user, item, context):
        score = combine(
            user_tower(user),
            item_tower(item),
            context_tower(context)
        )
```

### Multi-Objective Ranking

```python
# Optimize for multiple goals
def multi_objective_score(user, item):
    engagement_score = model_1(user, item)
    diversity_score = model_2(user, item)
    revenue_score = model_3(user, item)

    final_score = (
        0.6 * engagement_score +
        0.3 * diversity_score +
        0.1 * revenue_score
    )
```

### Online Retraining

```python
# Continuous learning from live traffic
@streaming_job
def online_train():
    for batch in stream_interactions():
        model.partial_fit(batch)
        if batch_count % 1000 == 0:
            evaluate_and_maybe_promote()
```

## Success Criteria

You've successfully completed the capstone if you can:

- ✅ Generate synthetic feed data
- ✅ Extract user and item features
- ✅ Train two-tower ranking model
- ✅ Generate personalized rankings
- ✅ Monitor feed quality
- ✅ Deploy end-to-end pipeline
- ✅ Explain the architecture

## Next Level

**Advanced Topics to Explore:**

1. **Graph Neural Networks** - Model user-item graph
2. **Transformers** - Sequence modeling for feeds
3. **Multi-Task Learning** - Predict clicks, likes, shares together
4. **Debiasing** - Address position bias, popularity bias
5. **Cold Start** - Handle new users/items
6. **Federated Learning** - Privacy-preserving recommendations

## Congratulations!

**You've completed the MLOps Mastery course!** 🎉

You now have:
- ✅ Production-ready MLOps system
- ✅ End-to-end pipeline experience
- ✅ Real-world architecture knowledge
- ✅ Portfolio-worthy project

**Your MLOps journey doesn't end here** - keep building, experimenting, and deploying ML systems!

---

## Resources

- [Two-Tower Models Paper](https://arxiv.org/abs/1606.07792)
- [Recommendations at Scale](https://developers.google.com/machine-learning/recommendation)
- [Production ML Best Practices](https://ml-ops.org/)
