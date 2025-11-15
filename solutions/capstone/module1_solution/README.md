# Capstone Module 1: Problem Definition & System Design

## Overview

This module defines the **Feed Ranking System** problem and designs the complete architecture.

## What You're Building

A personalized content feed ranking system that:
- Ranks items (posts, videos, articles) for each user
- Learns from user interactions (views, likes, shares)
- Updates daily with new data
- Scales to millions of users and items

## System Requirements

### Functional Requirements

1. **Personalization**: Different users see different item rankings
2. **Real-time Serving**: API responds in < 100ms
3. **Batch Updates**: Daily ranking pre-computation
4. **Quality**: Relevant, diverse, fresh content

### Non-Functional Requirements

1. **Scale**: 1M+ users, 10M+ items
2. **Latency**: P95 < 100ms for API
3. **Availability**: 99.9% uptime
4. **Freshness**: Rankings updated daily

## Architecture Design

### High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Users     │────>│ Ranking API  │<────│   Models    │
│             │     │  (FastAPI)   │     │ (Two-Tower) │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                    ▲
                            ▼                    │
                    ┌──────────────┐     ┌─────────────┐
                    │  Rankings    │     │  Training   │
                    │   Cache      │     │  Pipeline   │
                    │  (Redis)     │     │  (Airflow)  │
                    └──────────────┘     └─────────────┘
                                               ▲
                                               │
                                    ┌──────────────────┐
                                    │ User Activity &  │
                                    │  Item Catalog    │
                                    └──────────────────┘
```

### Components

1. **Data Pipeline** (Airflow)
   - User activity ingestion
   - Item catalog updates
   - Feature engineering
   - Training data generation

2. **ML Model** (PyTorch Two-Tower)
   - User embedding tower
   - Item embedding tower
   - Dot product scoring

3. **Serving Layer** (FastAPI)
   - Real-time ranking API
   - Batch pre-computed rankings
   - Redis caching

4. **Monitoring** (Custom)
   - Drift detection
   - Performance metrics
   - Quality monitoring

5. **Retraining** (Airflow)
   - Automated model updates
   - A/B testing
   - Safe deployment

## Data Flow

```
User Activity → Features → Training Pairs → Model → Rankings → API
     ↓                                                    ↓
Item Catalog                                        User Feeds
```

### Daily Schedule

```
00:00 - Collect user activity (past 24h)
01:00 - Update item catalog
02:00 - Extract features
03:00 - Generate training pairs
04:00 - Train/update model
05:00 - Generate rankings (batch)
06:00 - Update cache
```

## ML Problem Formulation

### Problem Type

**Binary Classification** (per user-item pair):
- Label = 1: User will engage with item
- Label = 0: User won't engage

### Features

**User Features:**
- Demographics (age, country, platform)
- Behavior (total interactions, like rate)
- Preferences (categories, creators)

**Item Features:**
- Metadata (category, tags, creator)
- Content (text embeddings, image features)
- Stats (age, popularity, quality score)

### Model Architecture

**Two-Tower Neural Network:**
```
User Features → User Tower → User Embedding (64-dim)
                                    ↓
                            Dot Product Score
                                    ↑
Item Features → Item Tower → Item Embedding (64-dim)
```

**Why Two-Tower?**
- ✅ Scalable: Embeddings computed independently
- ✅ Fast: Item embeddings pre-computed
- ✅ Flexible: Easy to update features

### Metrics

**Offline Metrics:**
- AUC-ROC: Model discrimination
- NDCG@k: Ranking quality
- Coverage: % items recommended

**Online Metrics:**
- CTR: Click-through rate
- Engagement: Likes, shares, comments
- Dwell Time: Time spent per item
- Retention: User retention rate

## Success Criteria

### Phase 1 (MVP)
- ✅ Basic two-tower model training
- ✅ Batch ranking generation
- ✅ Simple API serving

### Phase 2 (Production)
- ✅ Automated daily pipeline
- ✅ Monitoring and alerting
- ✅ A/B testing framework

### Phase 3 (Optimization)
- ✅ Real-time features
- ✅ Online learning
- ✅ Advanced ranking algorithms

## Key Decisions

### 1. Why Two-Tower vs. Single Model?

**Two-Tower Wins:**
- Faster inference (pre-compute item embeddings)
- Easier to scale (parallel computation)
- Better for production (update user/item separately)

### 2. Why Batch + Real-time Hybrid?

**Hybrid Approach:**
- Batch: Pre-compute top N candidates daily
- Real-time: Re-rank based on fresh context
- Best of both: Fast + fresh

### 3. Why Daily Retraining?

**Balance:**
- More frequent: Captures trends quickly
- Less frequent: More stable, lower cost
- Daily: Good balance for most feeds

## Dependencies on Previous Labs

```
Phase 1: Airflow basics → Pipeline orchestration
Phase 2: Feature engineering → User/item features
Phase 3: PyTorch training → Two-tower model
Phase 4: Serving + monitoring → Production system
```

## Next Steps

1. **Module 2**: Set up data collection and storage
2. **Module 3**: Build feature engineering pipeline
3. **Module 4**: Implement two-tower model
4. **Module 5**: Create ranking service
5. **Module 6**: Add monitoring and testing
6. **Module 7**: Integrate everything

## Key Learnings

After completing this module, you understand:
- ✅ How to formulate ranking as ML problem
- ✅ Two-tower architecture benefits
- ✅ Production ML system components
- ✅ Trade-offs in system design
- ✅ Metrics for success measurement
