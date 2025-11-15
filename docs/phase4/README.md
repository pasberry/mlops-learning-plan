# Phase 4: Deployment, Monitoring & Advanced MLOps

**Duration**: 2-3 weeks
**Goal**: Deploy models to production, implement monitoring, and close the MLOps loop

---

## Overview

Welcome to Phase 4 - the culmination of your MLOps journey! This is where everything comes together into a complete, production-ready ML system.

You've built data pipelines (Phase 2) and training workflows (Phase 3). Now you'll:
- **Deploy** models for online and batch inference
- **Monitor** model performance and data drift
- **Automate** retraining when drift is detected
- **Close the loop** with a complete MLOps system

### What You'll Learn

1. **Model Serving Patterns**
   - Online inference with FastAPI
   - Batch inference with Airflow
   - Model versioning and loading
   - Request/response validation

2. **Batch Processing**
   - Scheduled batch scoring
   - Large-scale inference pipelines
   - Output versioning and storage
   - Integration with downstream systems

3. **Monitoring & Observability**
   - Feature distribution tracking
   - Prediction distribution monitoring
   - Data drift detection (PSI, KL divergence)
   - Alerting and logging patterns

4. **Automated Retraining**
   - Drift-triggered retraining
   - Model comparison and validation
   - Automated promotion to production
   - A/B testing concepts

5. **Complete MLOps System**
   - End-to-end integration
   - Production best practices
   - Scaling considerations
   - Operational excellence

---

## The Complete MLOps Loop

By the end of Phase 4, you'll have implemented this entire loop:

```
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION ML SYSTEM                       │
│                                                             │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │   DATA   │──▶   │  TRAIN   │──▶   │  DEPLOY  │         │
│  │ Pipeline │      │ Pipeline │      │  Model   │         │
│  └──────────┘      └──────────┘      └────┬─────┘         │
│       ▲                                    │               │
│       │                                    ▼               │
│  ┌────┴──────┐                      ┌──────────┐          │
│  │ RETRAIN   │◀─────────────────────│ MONITOR  │          │
│  │ Pipeline  │    (drift detected)  │  & Alert │          │
│  └───────────┘                      └──────────┘          │
│                                            ▲               │
│                                            │               │
│                                      ┌──────────┐          │
│                                      │ PREDICT  │          │
│                                      │ (Online/ │          │
│                                      │  Batch)  │          │
│                                      └──────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Serving: Online vs Batch

### Online Inference (Real-time)
**Use Case**: Low-latency predictions for individual requests
- User clicks → recommendation
- Transaction → fraud score
- Query → search ranking

**Characteristics**:
- Latency: < 100ms typical
- Throughput: Hundreds to thousands of QPS
- Deployment: REST API (FastAPI, Flask)
- Scaling: Horizontal (multiple instances)

**Example**:
```python
# Client request
POST /predict
{
  "user_id": "12345",
  "features": [0.5, 1.2, ...]
}

# Response (50ms)
{
  "prediction": 0.87,
  "model_version": "v1.2.3"
}
```

### Batch Inference (Offline)
**Use Case**: High-throughput scoring of large datasets
- Nightly product recommendations
- Daily customer churn scores
- Weekly campaign targeting

**Characteristics**:
- Latency: Minutes to hours
- Throughput: Millions of predictions
- Deployment: Airflow DAG, Spark job
- Scaling: Vertical (bigger machines) or distributed

**Example**:
```python
# Airflow DAG runs daily at 2am
1. Read unscored data (1M users)
2. Load model
3. Generate predictions
4. Write to database/data lake
5. Notify downstream systems
```

---

## Monitoring: Why Models Degrade

Unlike traditional software, ML models degrade over time:

### 1. Data Drift
**The input distribution changes**

```
Training data (2023):          Production data (2024):
Age: 25-45 (mean=35)    →     Age: 18-30 (mean=24)
Income: $50k-$100k      →     Income: $30k-$70k
```

**Detection**: Compare feature distributions (PSI, KL divergence)

### 2. Concept Drift
**The relationship between X and Y changes**

```
2023: Price is #1 factor   →   2024: Reviews matter most
(price-sensitive users)        (quality-conscious users)
```

**Detection**: Monitor prediction accuracy if labels available

### 3. Prediction Drift
**The output distribution changes**

```
Training predictions:          Production predictions:
Class 0: 60%, Class 1: 40%  →  Class 0: 80%, Class 1: 20%
```

**Detection**: Compare prediction distributions

---

## Drift Detection Techniques

### Population Stability Index (PSI)
Measures change in distribution between two datasets.

```python
def calculate_psi(expected, actual, buckets=10):
    """
    PSI = Σ (actual% - expected%) * ln(actual% / expected%)

    PSI < 0.1: No significant change
    0.1 < PSI < 0.2: Moderate change, investigate
    PSI > 0.2: Significant change, retrain
    """
    pass
```

### Kullback-Leibler (KL) Divergence
Information theory metric for distribution comparison.

```python
def calculate_kl_divergence(p, q):
    """
    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

    Measures how much information is lost when Q
    is used to approximate P.
    """
    pass
```

### Statistical Tests
- **Kolmogorov-Smirnov test**: Compare continuous distributions
- **Chi-squared test**: Compare categorical distributions

---

## Retraining Strategies

### 1. Scheduled Retraining
**Trigger**: Time-based (daily, weekly, monthly)
```
Every Monday at 2am:
  - Gather last week's data
  - Retrain model
  - Validate on hold-out set
  - Deploy if better than current
```

**Pros**: Predictable, simple
**Cons**: May retrain unnecessarily or miss urgent drifts

### 2. Performance-Based Retraining
**Trigger**: Model accuracy drops below threshold
```
If model_accuracy < 0.85:
  - Alert team
  - Trigger retraining
  - Review data quality
```

**Pros**: Directly tied to business metric
**Cons**: Requires labeled data in production

### 3. Drift-Based Retraining
**Trigger**: Data drift exceeds threshold
```
If PSI > 0.2 for key features:
  - Log drift metrics
  - Trigger retraining
  - Compare new vs old model
```

**Pros**: Proactive, catches issues early
**Cons**: Drift doesn't always mean performance loss

### 4. Hybrid Approach (Recommended)
Combine multiple triggers:
```
Retrain if:
  - Weekly schedule (baseline)
  OR
  - PSI > 0.2 (drift detected)
  OR
  - Accuracy < 0.85 (performance degradation)
```

---

## Scaling Considerations

### Model Serving
**Problem**: 1000 requests/sec, 100ms latency requirement

**Solutions**:
1. **Horizontal scaling**: Multiple API instances + load balancer
2. **Model optimization**: ONNX, TorchScript, quantization
3. **Caching**: Cache predictions for common inputs
4. **Batching**: Accumulate requests, infer in batches
5. **GPU inference**: For complex models

### Batch Inference
**Problem**: Score 100M users in 2 hours

**Solutions**:
1. **Distributed processing**: Spark, Dask, Ray
2. **Data partitioning**: Process in chunks
3. **Parallelization**: Multiple workers
4. **Efficient data access**: Columnar formats (Parquet)
5. **Incremental scoring**: Only score new/changed entities

### Monitoring
**Problem**: Track millions of predictions per day

**Solutions**:
1. **Sampling**: Monitor 1% of traffic
2. **Aggregation**: Compute statistics, not individual records
3. **Time-series databases**: InfluxDB, Prometheus
4. **Alerting thresholds**: Focus on actionable metrics

---

## Phase 4 Labs

### Lab 4.1: Model Serving with FastAPI
**Goal**: Build a production-ready inference API
- Create FastAPI endpoints
- Load PyTorch models efficiently
- Implement request/response validation
- Add health checks and versioning

[→ Go to Lab 4.1](./lab4_1_model_serving.md)

### Lab 4.2: Batch Inference Pipeline
**Goal**: Implement scheduled batch scoring
- Create Airflow DAG for batch inference
- Read unscored data from storage
- Generate and write predictions
- Version outputs by date

[→ Go to Lab 4.2](./lab4_2_batch_inference.md)

### Lab 4.3: Monitoring & Drift Detection
**Goal**: Track model health and detect drift
- Implement feature distribution tracking
- Calculate PSI and KL divergence
- Create alerting logic
- Visualize drift over time

[→ Go to Lab 4.3](./lab4_3_monitoring.md)

### Lab 4.4: Automated Retraining Pipeline
**Goal**: Close the MLOps loop with automated retraining
- Create drift-triggered retraining DAG
- Compare new vs production models
- Implement model promotion logic
- Understand A/B testing concepts

[→ Go to Lab 4.4](./lab4_4_retraining_pipeline.md)

### Lab 4.5: Complete MLOps System
**Goal**: Integrate all components into one system
- Connect all pipelines end-to-end
- Implement the complete MLOps loop
- Production best practices
- Scaling and operational considerations

[→ Go to Lab 4.5](./lab4_5_complete_system.md)

---

## Success Criteria

You've completed Phase 4 when you can:

✅ Deploy models for both online and batch inference
✅ Implement production-grade monitoring and alerting
✅ Detect and quantify data drift
✅ Automate the retraining and deployment cycle
✅ Explain tradeoffs in serving and scaling strategies
✅ Operate a complete MLOps system end-to-end

---

## Production Best Practices

### 1. Model Versioning
```
models/
  production/
    model_v1.0.0.pt
    model_v1.1.0.pt
    model_v2.0.0.pt  ← current
  staging/
    model_v2.1.0-candidate.pt
```

**Semantic versioning**:
- Major: Breaking changes (new features, architecture)
- Minor: Backward-compatible improvements
- Patch: Bug fixes, retraining on new data

### 2. Gradual Rollouts
Don't deploy to 100% immediately:
```
1. Deploy to staging
2. A/B test on 5% of traffic
3. Monitor for 24-48 hours
4. Ramp to 25% → 50% → 100%
5. Keep old model warm (instant rollback)
```

### 3. Monitoring Everything
```
Application metrics:
  - Request latency (p50, p95, p99)
  - Error rate
  - Throughput (QPS)

Model metrics:
  - Prediction distribution
  - Feature distributions
  - Drift scores

Business metrics:
  - Click-through rate
  - Conversion rate
  - Revenue impact
```

### 4. Graceful Degradation
When things fail:
```python
try:
    prediction = model.predict(features)
except Exception as e:
    log_error(e)
    # Fallback to rule-based system
    prediction = fallback_predictor(features)
```

### 5. Documentation
Maintain a model card for each production model:
```yaml
model_id: user_churn_v2.1.0
training_date: 2024-01-15
training_data: users_2023_q4
features: [age, tenure, usage, ...]
metrics:
  - auc: 0.87
  - precision: 0.82
  - recall: 0.79
deployed_to: production
deployment_date: 2024-01-20
```

---

## Common Pitfalls

### Serving
❌ **Loading model on every request**: Load once, reuse
❌ **No request validation**: Validate input schema
❌ **Synchronous preprocessing**: Can be slow, consider async
❌ **No timeout handling**: Set reasonable timeouts

### Monitoring
❌ **Alert fatigue**: Too many noisy alerts
❌ **Monitoring without action**: Alerts should be actionable
❌ **No baseline**: Track metrics from day one
❌ **Ignoring data quality**: GIGO (Garbage In, Garbage Out)

### Retraining
❌ **Automatic deployment without validation**: Always validate first
❌ **No rollback plan**: Keep previous model ready
❌ **Retraining on bad data**: Data quality checks first
❌ **No A/B testing**: Compare before full rollout

---

## Next Steps

1. **Set up FastAPI** for model serving (Lab 4.1)
2. **Complete all labs** in sequence
3. **Build your portfolio**: This is your capstone project
4. **Share your work**: Document your complete MLOps system
5. **Keep learning**: Production ML is constantly evolving

---

## Real-World Production Architectures

### Startup (< 1M requests/day)
```
FastAPI → Model (on same instance)
Batch scoring: Single Airflow worker
Monitoring: CloudWatch/Datadog
Database: PostgreSQL
```

### Mid-size (1M-100M requests/day)
```
Load Balancer → [FastAPI instances] → Redis cache
Batch scoring: Airflow with Celery executor
Monitoring: Prometheus + Grafana
Database: PostgreSQL + S3 for features
```

### Large-scale (100M+ requests/day)
```
CDN → Load Balancer → [API instances] → Feature Store
Model serving: TorchServe/Triton on Kubernetes
Batch scoring: Spark on EMR/Databricks
Monitoring: Prometheus + custom dashboards
Storage: S3/GCS + Snowflake/BigQuery
```

---

## Resources

### Model Serving
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TorchServe](https://pytorch.org/serve/)
- [ONNX Runtime](https://onnxruntime.ai/)

### Monitoring
- [Evidently AI](https://www.evidentlyai.com/)
- [WhyLabs](https://whylabs.ai/)
- [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/en/stable/)

### MLOps Best Practices
- [Google MLOps](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [AWS MLOps](https://aws.amazon.com/sagemaker/mlops/)
- [MLOps.org](https://ml-ops.org/)

### Research Papers
- [Sculley et al. - Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- [Breck et al. - ML Test Score](https://research.google/pubs/pub46555/)

---

**Ready to deploy to production? Let's build a complete MLOps system!** 🚀
