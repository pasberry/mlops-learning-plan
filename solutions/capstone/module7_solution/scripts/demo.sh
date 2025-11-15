#!/bin/bash

# Feed Ranking System - Demo Script
# Demonstrates the complete end-to-end system

set -e  # Exit on error

echo "============================================================"
echo "Feed Ranking System - Complete Demo"
echo "============================================================"
echo ""

# Configuration
BASE_DIR="/home/user/mlops-learning-plan"
DATA_DIR="$BASE_DIR/data/feed"
MODELS_DIR="$BASE_DIR/models/feed_ranking"
AIRFLOW_HOME="${AIRFLOW_HOME:-$BASE_DIR/airflow}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Setup directories
print_step "Step 1: Setting up directories"
mkdir -p "$DATA_DIR"/{raw,processed,reports}
mkdir -p "$MODELS_DIR"
echo "✓ Directories created"
echo ""

# Step 2: Generate synthetic data
print_step "Step 2: Generating synthetic feed data"

python3 << 'PYTHON_EOF'
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(42)

print("Generating synthetic data...")

# Generate users
n_users = 1000
users = pd.DataFrame({
    'user_id': range(1, n_users + 1),
    'user_age': np.random.randint(18, 65, n_users),
    'user_country': np.random.choice(['US', 'UK', 'CA', 'AU'], n_users),
    'user_platform': np.random.choice(['ios', 'android', 'web'], n_users),
})

# Generate items (content)
n_items = 5000
items = pd.DataFrame({
    'item_id': range(1, n_items + 1),
    'item_category': np.random.choice(['tech', 'sports', 'news', 'entertainment'], n_items),
    'item_age_hours': np.random.exponential(24, n_items),
    'creator_followers': np.random.lognormal(8, 2, n_items).astype(int),
})

# Generate user-item interactions
n_interactions = 50000
user_ids = np.random.choice(users['user_id'], n_interactions)
item_ids = np.random.choice(items['item_id'], n_interactions)

interactions = pd.DataFrame({
    'user_id': user_ids,
    'item_id': item_ids,
    'timestamp': [
        (datetime.now() - timedelta(hours=np.random.randint(0, 72))).isoformat()
        for _ in range(n_interactions)
    ],
    'interaction_type': np.random.choice(['view', 'like', 'share', 'comment'], n_interactions),
    'dwell_time': np.random.lognormal(2, 1, n_interactions)
})

# Save data
base_path = Path('/home/user/mlops-learning-plan/data/feed')
interactions.to_csv(base_path / 'user_activity.csv', index=False)
items.to_csv(base_path / 'items.csv', index=False)
users.to_csv(base_path / 'users.csv', index=False)

print(f"✓ Generated {len(users)} users")
print(f"✓ Generated {len(items)} items")
print(f"✓ Generated {len(interactions)} interactions")
PYTHON_EOF

echo "✓ Data generation complete"
echo ""

# Step 3: Test feature extraction
print_step "Step 3: Testing feature extraction"

python3 << 'PYTHON_EOF'
import pandas as pd
from pathlib import Path

# Load data
base_path = Path('/home/user/mlops-learning-plan/data/feed')
activity = pd.read_csv(base_path / 'user_activity.csv')

# Extract user features
user_features = activity.groupby('user_id').agg({
    'timestamp': 'count',
    'item_id': 'nunique',
    'interaction_type': lambda x: (x == 'like').sum(),
    'dwell_time': 'mean'
}).reset_index()

user_features.columns = ['user_id', 'total_interactions', 'unique_items', 'total_likes', 'avg_dwell_time']
user_features['like_rate'] = user_features['total_likes'] / user_features['total_interactions']

user_features.to_csv(base_path / 'user_features.csv', index=False)

print(f"✓ Extracted features for {len(user_features)} users")
print(f"  Avg interactions per user: {user_features['total_interactions'].mean():.1f}")

# Extract item features
items = pd.read_csv(base_path / 'items.csv')
item_features = items.copy()
item_features.to_csv(base_path / 'item_features.csv', index=False)

print(f"✓ Extracted features for {len(item_features)} items")
PYTHON_EOF

echo "✓ Feature extraction complete"
echo ""

# Step 4: Copy DAG to Airflow
print_step "Step 4: Deploying DAG to Airflow"

if [ -z "$AIRFLOW_HOME" ]; then
    print_warning "AIRFLOW_HOME not set. Skipping DAG deployment."
    print_warning "Set AIRFLOW_HOME and run: cp dags/feed_master_dag.py \$AIRFLOW_HOME/dags/"
else
    mkdir -p "$AIRFLOW_HOME/dags"
    cp dags/feed_master_dag.py "$AIRFLOW_HOME/dags/"
    echo "✓ DAG deployed to $AIRFLOW_HOME/dags/"
fi
echo ""

# Step 5: Summary
print_step "Step 5: System Ready"

echo "============================================================"
echo "Feed Ranking System Demo Complete!"
echo "============================================================"
echo ""
echo "Data generated:"
echo "  • Users: $DATA_DIR/users.csv"
echo "  • Items: $DATA_DIR/items.csv"
echo "  • Interactions: $DATA_DIR/user_activity.csv"
echo "  • User Features: $DATA_DIR/user_features.csv"
echo "  • Item Features: $DATA_DIR/item_features.csv"
echo ""
echo "Next steps:"
echo "  1. Start Airflow:"
echo "     airflow webserver --port 8080"
echo "     airflow scheduler"
echo ""
echo "  2. Trigger the DAG:"
echo "     airflow dags trigger feed_ranking_master"
echo ""
echo "  3. Monitor at http://localhost:8080"
echo ""
echo "  4. View results in:"
echo "     $DATA_DIR/rankings.csv"
echo "     $DATA_DIR/reports/system_report.json"
echo ""
echo "============================================================"
