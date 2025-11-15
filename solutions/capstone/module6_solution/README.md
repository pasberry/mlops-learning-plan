# Capstone Module 6: Testing, Monitoring & Experimentation

## Overview

Ensures the feed ranking system is reliable, monitored, and continuously improving through testing and experimentation.

## Testing Strategy

### Unit Tests

```python
# tests/test_model.py
def test_two_tower_model():
    """Test model forward pass."""
    model = TwoTowerModel(user_dim=10, item_dim=8, embedding_dim=64)

    user_features = torch.randn(32, 10)
    item_features = torch.randn(32, 8)

    output = model(user_features, item_features)

    assert output.shape == (32, 1)
    assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output

def test_feature_engineering():
    """Test feature extraction."""
    activity_df = create_test_activity_data()
    user_features = extract_user_features(activity_df)

    assert 'like_rate' in user_features.columns
    assert user_features['like_rate'].between(0, 1).all()
    assert not user_features.isnull().any().any()
```

### Integration Tests

```python
# tests/test_pipeline.py
def test_end_to_end_pipeline():
    """Test complete ranking pipeline."""
    # 1. Generate test data
    test_users = [1, 2, 3]
    test_items = list(range(100))

    # 2. Extract features
    user_features = extract_features(test_users)
    item_features = extract_features(test_items)

    # 3. Load model
    model = load_model('models/test_model.pt')

    # 4. Generate rankings
    for user_id in test_users:
        rankings = generate_rankings(model, user_id, item_features)

        assert len(rankings) > 0
        assert all('item_id' in r for r in rankings)
        assert all('score' in r for r in rankings)
        assert rankings[0]['score'] >= rankings[-1]['score']  # Sorted

def test_api_endpoints():
    """Test ranking API."""
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # Health check
    response = client.get("/health")
    assert response.status_code == 200

    # Ranking request
    response = client.post("/rank", json={
        "user_id": 1,
        "num_items": 50
    })

    assert response.status_code == 200
    data = response.json()
    assert data['user_id'] == 1
    assert len(data['items']) == 50
    assert data['latency_ms'] > 0
```

### Load Tests

```python
# tests/load_test.py
from locust import HttpUser, task, between

class RankingUser(HttpUser):
    """Simulate user load on ranking API."""

    wait_time = between(1, 3)

    @task
    def get_feed(self):
        """Request personalized feed."""
        user_id = random.randint(1, 10000)

        self.client.post("/rank", json={
            "user_id": user_id,
            "num_items": 50
        })

# Run: locust -f tests/load_test.py --host=http://localhost:8000
```

## Monitoring

### Performance Metrics

```python
class PerformanceMonitor:
    """Monitor system performance."""

    def __init__(self):
        self.metrics = {
            'latency_p50': Histogram('latency_p50'),
            'latency_p95': Histogram('latency_p95'),
            'latency_p99': Histogram('latency_p99'),
            'throughput': Gauge('throughput_rps'),
            'error_rate': Counter('errors_total')
        }

    def track_request(self, latency_ms, success):
        """Track request metrics."""
        self.metrics['latency_p50'].observe(latency_ms)
        self.metrics['latency_p95'].observe(latency_ms)
        self.metrics['latency_p99'].observe(latency_ms)

        if not success:
            self.metrics['error_rate'].inc()

    def get_summary(self):
        """Get performance summary."""
        return {
            'p50': self.metrics['latency_p50'].get_percentile(50),
            'p95': self.metrics['latency_p95'].get_percentile(95),
            'p99': self.metrics['latency_p99'].get_percentile(99),
            'throughput': self.metrics['throughput'].get(),
            'error_rate': self.metrics['error_rate'].get()
        }
```

### Quality Metrics

```python
class QualityMonitor:
    """Monitor ranking quality."""

    def track_ranking_quality(self, user_id, rankings, interactions):
        """
        Track quality metrics per ranking session.
        """
        # Diversity
        unique_categories = len(set(r['category'] for r in rankings))
        category_entropy = entropy([...])  # Category distribution

        # Coverage
        items_shown = set(r['item_id'] for r in rankings)
        catalog_coverage = len(items_shown) / total_catalog_size

        # Freshness
        avg_item_age_hours = mean(r['age_hours'] for r in rankings)

        # Relevance (if we have interactions)
        if interactions:
            ctr = len(interactions) / len(rankings)
            engagement_rate = sum(i['engaged'] for i in interactions) / len(interactions)
        else:
            ctr = None
            engagement_rate = None

        return {
            'diversity': {
                'unique_categories': unique_categories,
                'category_entropy': category_entropy
            },
            'coverage': catalog_coverage,
            'freshness': avg_item_age_hours,
            'relevance': {
                'ctr': ctr,
                'engagement_rate': engagement_rate
            }
        }
```

### Drift Monitoring

```python
from solutions.phase4.lab4_3_solution.monitoring.drift_detector import DriftDetector

class RankingDriftMonitor:
    """Monitor drift in rankings and features."""

    def __init__(self):
        self.drift_detector = DriftDetector(
            psi_threshold=0.2,
            ks_threshold=0.05
        )

    def check_feature_drift(self, baseline_features, current_features):
        """Check for drift in input features."""
        drift_report = self.drift_detector.detect_feature_drift(
            baseline_features,
            current_features,
            feature_columns=['age', 'like_rate', 'total_interactions']
        )

        if drift_report['overall_drift_detected']:
            alert_team(f"Feature drift detected: {drift_report}")

        return drift_report

    def check_score_drift(self, baseline_scores, current_scores):
        """Check for drift in model scores."""
        drift_report = self.drift_detector.detect_prediction_drift(
            baseline_scores,
            current_scores
        )

        if drift_report['drift_detected']:
            alert_team(f"Score drift detected: {drift_report}")

        return drift_report
```

## A/B Testing

### Experiment Framework

```python
class ABTestingFramework:
    """A/B testing for ranking algorithms."""

    def __init__(self):
        self.experiments = {}
        self.assignment_cache = {}

    def create_experiment(self, experiment_id, config):
        """
        Create new A/B test.

        config = {
            'name': 'test_new_ranking_v2',
            'variants': [
                {'id': 'control', 'weight': 0.5, 'model': 'v1'},
                {'id': 'treatment', 'weight': 0.5, 'model': 'v2'}
            ],
            'metrics': ['ctr', 'engagement_rate', 'dwell_time'],
            'start_date': '2025-11-15',
            'end_date': '2025-11-30'
        }
        """
        self.experiments[experiment_id] = config

    def assign_variant(self, user_id, experiment_id):
        """
        Assign user to experiment variant.
        Uses consistent hashing for stable assignments.
        """
        # Check cache
        cache_key = f"{user_id}:{experiment_id}"
        if cache_key in self.assignment_cache:
            return self.assignment_cache[cache_key]

        # Hash user_id for consistent assignment
        hash_val = hash(f"{user_id}:{experiment_id}") % 100

        # Assign based on weights
        experiment = self.experiments[experiment_id]
        cumulative_weight = 0

        for variant in experiment['variants']:
            cumulative_weight += variant['weight'] * 100
            if hash_val < cumulative_weight:
                self.assignment_cache[cache_key] = variant['id']
                return variant['id']

    def track_metric(self, user_id, experiment_id, metric_name, value):
        """Track experiment metric."""
        variant = self.assign_variant(user_id, experiment_id)

        # Store metric
        metric_key = f"experiment:{experiment_id}:{variant}:{metric_name}"
        redis.lpush(metric_key, value)

    def analyze_results(self, experiment_id):
        """Analyze A/B test results."""
        experiment = self.experiments[experiment_id]
        results = {}

        for variant in experiment['variants']:
            variant_id = variant['id']
            variant_metrics = {}

            for metric_name in experiment['metrics']:
                # Get all values
                metric_key = f"experiment:{experiment_id}:{variant_id}:{metric_name}"
                values = redis.lrange(metric_key, 0, -1)

                # Calculate statistics
                variant_metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }

            results[variant_id] = variant_metrics

        # Statistical significance test
        control_ctr = results['control']['ctr']['mean']
        treatment_ctr = results['treatment']['ctr']['mean']

        # T-test
        p_value = ttest_ind(
            redis.lrange(f"experiment:{experiment_id}:control:ctr", 0, -1),
            redis.lrange(f"experiment:{experiment_id}:treatment:ctr", 0, -1)
        ).pvalue

        results['statistical_significance'] = {
            'p_value': p_value,
            'significant': p_value < 0.05,
            'relative_lift': (treatment_ctr - control_ctr) / control_ctr * 100
        }

        return results
```

### Experiment Integration

```python
@app.post("/rank")
async def rank_feed(request: RankingRequest):
    """Generate rankings with A/B testing."""

    # Check for active experiments
    active_experiments = get_active_experiments(request.user_id)

    if active_experiments:
        # Assign to variant
        variant = ab_framework.assign_variant(
            request.user_id,
            active_experiments[0]['id']
        )

        # Use appropriate model
        if variant == 'treatment':
            model = load_model('models/v2/model.pt')
        else:
            model = load_model('models/v1/model.pt')
    else:
        # Default model
        model = load_model('models/production/model.pt')

    # Generate rankings
    rankings = generate_rankings(model, request.user_id)

    # Track experiment assignment
    if active_experiments:
        track_experiment_exposure(
            request.user_id,
            active_experiments[0]['id'],
            variant
        )

    return rankings
```

## Alerting

### Alert Rules

```python
class AlertingSystem:
    """Alert on system issues."""

    def __init__(self):
        self.rules = [
            {'name': 'high_latency', 'condition': lambda m: m['p95'] > 100, 'severity': 'warning'},
            {'name': 'high_error_rate', 'condition': lambda m: m['error_rate'] > 0.01, 'severity': 'critical'},
            {'name': 'low_cache_hit', 'condition': lambda m: m['cache_hit_rate'] < 0.8, 'severity': 'warning'},
            {'name': 'model_drift', 'condition': lambda m: m['drift_detected'], 'severity': 'warning'},
        ]

    def check_alerts(self, metrics):
        """Check all alert rules."""
        triggered_alerts = []

        for rule in self.rules:
            if rule['condition'](metrics):
                alert = {
                    'name': rule['name'],
                    'severity': rule['severity'],
                    'metrics': metrics,
                    'timestamp': datetime.now()
                }
                triggered_alerts.append(alert)
                self.send_alert(alert)

        return triggered_alerts

    def send_alert(self, alert):
        """Send alert to on-call team."""
        if alert['severity'] == 'critical':
            # PagerDuty
            pagerduty.trigger(alert)
            # Slack
            slack.send_message('#oncall', f"🚨 CRITICAL: {alert['name']}")
        else:
            # Slack only
            slack.send_message('#ml-alerts', f"⚠️  WARNING: {alert['name']}")
```

## Dashboards

### Grafana Dashboard Config

```yaml
# grafana/dashboards/ranking_system.json
{
  "dashboard": {
    "title": "Feed Ranking System",
    "panels": [
      {
        "title": "Request Latency",
        "type": "graph",
        "targets": [
          {"expr": "histogram_quantile(0.50, ranking_latency_seconds)"},
          {"expr": "histogram_quantile(0.95, ranking_latency_seconds)"},
          {"expr": "histogram_quantile(0.99, ranking_latency_seconds)"}
        ]
      },
      {
        "title": "Throughput",
        "type": "graph",
        "targets": [
          {"expr": "rate(ranking_requests_total[5m])"}
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {"expr": "rate(ranking_errors_total[5m])"}
        ]
      },
      {
        "title": "Model Performance",
        "type": "table",
        "targets": [
          {"expr": "model_auc"},
          {"expr": "model_ndcg"}
        ]
      }
    ]
  }
}
```

## Integration Points

```
Module 4 (Training) → Model evaluation
Module 5 (Serving) → Performance monitoring
Module 7 (System) → End-to-end testing
```

## Learning Outcomes

✅ Comprehensive testing strategies
✅ Production monitoring
✅ A/B testing frameworks
✅ Alerting and incident response
✅ Quality assurance
