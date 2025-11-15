# Module 2: ETL & Feature Engineering DAG

**Estimated Time**: 2-3 days
**Difficulty**: Medium-Hard

## Learning Objectives

By the end of this module, you will be able to:
- ✅ Generate realistic synthetic feed interaction data
- ✅ Build production-grade data validation pipelines with Great Expectations
- ✅ Engineer ranking-specific features from raw interaction data
- ✅ Create proper temporal train/val/test splits
- ✅ Orchestrate ETL workflows with Airflow
- ✅ Handle data quality issues and edge cases

## Overview

The ETL (Extract, Transform, Load) pipeline is the foundation of your ML system. This module builds a production-quality data pipeline that:
1. Generates realistic user-item interaction data
2. Validates data quality at every step
3. Engineers features used by the ranking model
4. Creates train/validation/test splits
5. Orchestrates everything with Airflow

**Key Principle**: Garbage in, garbage out. The quality of your model depends entirely on the quality of your data and features.

## Background

### Feed Ranking Data Patterns

Real feed ranking systems have specific data characteristics:

**User Behavior**:
- **Power law distribution**: 1% of users generate 50% of interactions
- **Temporal patterns**: More engagement on weekends, evenings
- **Cold start**: New users have no interaction history
- **Preference stability**: User interests change slowly over time

**Item Characteristics**:
- **Recency bias**: Newer items get more impressions
- **Popularity bias**: Popular items get more engagement
- **Content variety**: Different categories have different engagement rates
- **Lifecycle**: Items have a "shelf life" (viral phase → decay)

**Interaction Signals**:
- **Click**: Binary, easy to collect
- **Dwell time**: Continuous, indicates interest depth
- **Likes/Shares**: Explicit positive signals
- **Skips**: Negative signal (user saw but didn't engage)

### Feature Engineering for Ranking

Good ranking features:
- **User features**: Historical behavior, preferences, demographics
- **Item features**: Content properties, popularity, freshness
- **User-Item features**: Past interactions, affinity scores
- **Contextual features**: Time, device, location

## Step 1: Synthetic Data Generation

### Data Generation Strategy

We'll generate data that simulates realistic patterns.

**File**: `src/data/generator.py`

```python
"""
Synthetic feed interaction data generator.

Simulates realistic user-item interaction patterns for a feed ranking system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Tuple
import yaml


class FeedDataGenerator:
    """
    Generates synthetic feed interaction data with realistic patterns.

    Patterns simulated:
    - Power law user behavior (some users very active)
    - Item popularity distribution
    - Temporal patterns (time of day, day of week)
    - User-item affinity (users prefer certain categories)
    - Cold start (new users/items)
    """

    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Load configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_users = self.config['data_generation']['num_users']
        self.num_items = self.config['data_generation']['num_items']
        self.num_interactions = self.config['data_generation']['num_interactions']

        # Parse date range
        self.start_date = datetime.fromisoformat(
            self.config['data_generation']['date_range']['start']
        )
        self.end_date = datetime.fromisoformat(
            self.config['data_generation']['date_range']['end']
        )

        # Initialize random state for reproducibility
        np.random.seed(42)

        # Generate user and item metadata
        self._generate_metadata()

    def _generate_metadata(self):
        """Generate user and item metadata."""
        # User segments (power law distribution)
        # 10% power users, 30% regular, 60% casual
        user_segments = np.random.choice(
            ['power', 'regular', 'casual'],
            size=self.num_users,
            p=[0.1, 0.3, 0.6]
        )

        self.user_activity_level = {
            'power': 100,
            'regular': 20,
            'casual': 5
        }

        self.users = pd.DataFrame({
            'user_id': range(self.num_users),
            'segment': user_segments,
            'signup_date': [
                self.start_date + timedelta(days=np.random.randint(0, 300))
                for _ in range(self.num_users)
            ],
            'preferred_category': np.random.randint(0, 10, self.num_users),
            'device_preference': np.random.choice(
                ['mobile', 'desktop', 'tablet'],
                size=self.num_users,
                p=[0.6, 0.3, 0.1]
            )
        })

        # Item metadata
        item_creation_dates = [
            self.start_date + timedelta(days=np.random.randint(0, 320))
            for _ in range(self.num_items)
        ]

        self.items = pd.DataFrame({
            'item_id': range(self.num_items),
            'category': np.random.randint(0, 10, self.num_items),
            'creation_date': item_creation_dates,
            'base_quality': np.random.beta(5, 2, self.num_items)  # Skewed toward high quality
        })

    def generate_interactions(self) -> pd.DataFrame:
        """
        Generate interaction data.

        Returns:
            DataFrame with columns: interaction_id, user_id, item_id,
            timestamp, click, like, share, dwell_time, device, time_of_day, day_of_week
        """
        interactions = []

        print(f"Generating {self.num_interactions} interactions...")

        for i in range(self.num_interactions):
            if i % 100000 == 0:
                print(f"  Progress: {i}/{self.num_interactions}")

            # Sample user (weighted by activity level)
            user_weights = self.users['segment'].map(self.user_activity_level).values
            user_weights = user_weights / user_weights.sum()
            user_idx = np.random.choice(len(self.users), p=user_weights)
            user = self.users.iloc[user_idx]

            # Sample item (with recency and quality bias)
            item_idx = self._sample_item(user)
            item = self.items.iloc[item_idx]

            # Generate timestamp (with temporal patterns)
            timestamp = self._generate_timestamp(user)

            # Determine engagement (based on user-item affinity)
            engagement = self._determine_engagement(user, item, timestamp)

            # Device (mostly user's preference, but some variation)
            device = user['device_preference'] if np.random.random() < 0.8 else \
                     np.random.choice(['mobile', 'desktop', 'tablet'])

            # Time context
            hour = timestamp.hour
            if 5 <= hour < 12:
                time_of_day = 'morning'
            elif 12 <= hour < 17:
                time_of_day = 'afternoon'
            elif 17 <= hour < 22:
                time_of_day = 'evening'
            else:
                time_of_day = 'night'

            interaction = {
                'interaction_id': str(uuid.uuid4()),
                'user_id': int(user['user_id']),
                'item_id': int(item['item_id']),
                'timestamp': timestamp.isoformat(),
                'click': engagement['click'],
                'like': engagement['like'],
                'share': engagement['share'],
                'dwell_time': engagement['dwell_time'],
                'device': device,
                'time_of_day': time_of_day,
                'day_of_week': timestamp.weekday()
            }

            interactions.append(interaction)

        df = pd.DataFrame(interactions)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Generated {len(df)} interactions")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"CTR: {df['click'].mean():.3f}")

        return df

    def _sample_item(self, user: pd.Series) -> int:
        """Sample item with preference for user's category and recent items."""
        # Higher probability for user's preferred category
        category_match = (self.items['category'] == user['preferred_category']).astype(float)

        # Higher probability for recent items (recency bias)
        days_since_creation = (self.end_date - self.items['creation_date']).dt.days
        recency_score = 1.0 / (days_since_creation + 1)  # Avoid division by zero

        # Higher probability for high-quality items
        quality_score = self.items['base_quality']

        # Combine scores
        item_weights = (
            category_match * 3.0 +  # Category match is important
            recency_score * 2.0 +   # Recency matters
            quality_score * 1.0     # Quality helps
        )

        item_weights = item_weights / item_weights.sum()

        return np.random.choice(len(self.items), p=item_weights)

    def _generate_timestamp(self, user: pd.Series) -> datetime:
        """Generate timestamp with temporal patterns."""
        # Random day in range, but after user signup
        days_range = (self.end_date - user['signup_date']).days
        if days_range <= 0:
            random_date = user['signup_date']
        else:
            random_date = user['signup_date'] + timedelta(days=np.random.randint(0, days_range))

        # Hour of day (non-uniform: peaks in evening)
        # Use beta distribution to create realistic hourly pattern
        hour_prob = np.random.beta(2, 2)  # Centered distribution
        if random_date.weekday() < 5:  # Weekday
            # Peak in evening (18-22)
            hour = int(12 + hour_prob * 12)  # 12-24 hours
        else:  # Weekend
            # More uniform, but still favor daytime
            hour = int(8 + hour_prob * 16)  # 8-24 hours

        hour = min(hour, 23)

        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)

        return random_date.replace(hour=hour, minute=minute, second=second)

    def _determine_engagement(
        self,
        user: pd.Series,
        item: pd.Series,
        timestamp: datetime
    ) -> Dict[str, any]:
        """
        Determine engagement signals based on user-item affinity.

        Returns dict with: click, like, share, dwell_time
        """
        # Base engagement probability
        base_prob = item['base_quality']

        # Category match bonus
        if user['preferred_category'] == item['category']:
            base_prob *= 1.5

        # Recency bonus (newer items get more engagement)
        days_old = (timestamp - item['creation_date']).days
        recency_multiplier = 1.0 / (1.0 + days_old / 30.0)  # Decay over 30 days
        base_prob *= recency_multiplier

        # User segment affects engagement
        segment_multiplier = {
            'power': 1.5,
            'regular': 1.0,
            'casual': 0.7
        }
        base_prob *= segment_multiplier[user['segment']]

        # Clip probability
        click_prob = min(base_prob, 0.9)

        # Determine click
        click = int(np.random.random() < click_prob)

        # Given click, determine other signals
        if click:
            like = int(np.random.random() < 0.3)  # 30% of clicks get likes
            share = int(np.random.random() < 0.1 * (1 + like))  # Shares are rare, more likely if liked

            # Dwell time (exponential distribution, mean=30s)
            dwell_time = np.random.exponential(30)
            # Quality content gets longer dwell time
            dwell_time *= item['base_quality']
            # Clip at 300s (5 minutes)
            dwell_time = min(dwell_time, 300)
        else:
            like = 0
            share = 0
            dwell_time = 0.0

        return {
            'click': click,
            'like': like,
            'share': share,
            'dwell_time': round(dwell_time, 2)
        }


def main():
    """Generate data and save to CSV."""
    generator = FeedDataGenerator()

    # Generate interactions
    interactions_df = generator.generate_interactions()

    # Save to CSV
    output_path = "data/raw/interactions.csv"
    interactions_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Save metadata
    generator.users.to_csv("data/raw/users.csv", index=False)
    generator.items.to_csv("data/raw/items.csv", index=False)
    print("Saved user and item metadata")


if __name__ == "__main__":
    main()
```

**Run it**:
```bash
cd capstone_project
python src/data/generator.py
```

**Expected Output**:
- `data/raw/interactions.csv` (1M rows)
- `data/raw/users.csv` (10K rows)
- `data/raw/items.csv` (50K rows)

## Step 2: Data Validation

### Great Expectations Setup

**File**: `src/data/validator.py`

```python
"""
Data validation using Great Expectations.

Validates:
- Schema correctness
- Value ranges
- Null rates
- Statistical distributions
"""

import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
import pandas as pd
from pathlib import Path
import json


class DataValidator:
    """Validate data quality using Great Expectations."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.context = gx.get_context()

    def validate_interactions(self, file_path: str) -> dict:
        """
        Validate interaction data.

        Returns:
            dict with validation results
        """
        print(f"Validating {file_path}...")

        # Load data
        df = pd.read_csv(file_path)

        # Create expectations
        expectations = []

        # Schema expectations
        required_columns = [
            'interaction_id', 'user_id', 'item_id', 'timestamp',
            'click', 'like', 'share', 'dwell_time',
            'device', 'time_of_day', 'day_of_week'
        ]

        for col in required_columns:
            expectations.append({
                'expectation_type': 'expect_column_to_exist',
                'kwargs': {'column': col}
            })

        # Type expectations
        expectations.extend([
            # IDs should be non-null
            {
                'expectation_type': 'expect_column_values_to_not_be_null',
                'kwargs': {'column': 'interaction_id'}
            },
            {
                'expectation_type': 'expect_column_values_to_not_be_null',
                'kwargs': {'column': 'user_id'}
            },
            {
                'expectation_type': 'expect_column_values_to_not_be_null',
                'kwargs': {'column': 'item_id'}
            },

            # Binary columns should be 0 or 1
            {
                'expectation_type': 'expect_column_values_to_be_in_set',
                'kwargs': {'column': 'click', 'value_set': [0, 1]}
            },
            {
                'expectation_type': 'expect_column_values_to_be_in_set',
                'kwargs': {'column': 'like', 'value_set': [0, 1]}
            },
            {
                'expectation_type': 'expect_column_values_to_be_in_set',
                'kwargs': {'column': 'share', 'value_set': [0, 1]}
            },

            # Dwell time should be non-negative
            {
                'expectation_type': 'expect_column_values_to_be_between',
                'kwargs': {'column': 'dwell_time', 'min_value': 0, 'max_value': 300}
            },

            # Device should be in known set
            {
                'expectation_type': 'expect_column_values_to_be_in_set',
                'kwargs': {
                    'column': 'device',
                    'value_set': ['mobile', 'desktop', 'tablet']
                }
            },

            # Day of week should be 0-6
            {
                'expectation_type': 'expect_column_values_to_be_between',
                'kwargs': {'column': 'day_of_week', 'min_value': 0, 'max_value': 6}
            },
        ])

        # Statistical expectations
        expectations.extend([
            # CTR should be reasonable (5-30%)
            {
                'expectation_type': 'expect_column_mean_to_be_between',
                'kwargs': {'column': 'click', 'min_value': 0.05, 'max_value': 0.35}
            },

            # Most interactions should have no like (likes are rare)
            {
                'expectation_type': 'expect_column_mean_to_be_between',
                'kwargs': {'column': 'like', 'min_value': 0.01, 'max_value': 0.15}
            },
        ])

        # Run validation
        results = self._run_expectations(df, expectations)

        return results

    def _run_expectations(self, df: pd.DataFrame, expectations: list) -> dict:
        """Run expectations and return results."""
        results = {
            'total': len(expectations),
            'passed': 0,
            'failed': 0,
            'failures': []
        }

        for expectation in expectations:
            exp_type = expectation['expectation_type']
            kwargs = expectation['kwargs']

            try:
                if exp_type == 'expect_column_to_exist':
                    passed = kwargs['column'] in df.columns

                elif exp_type == 'expect_column_values_to_not_be_null':
                    passed = df[kwargs['column']].notna().all()

                elif exp_type == 'expect_column_values_to_be_in_set':
                    passed = df[kwargs['column']].isin(kwargs['value_set']).all()

                elif exp_type == 'expect_column_values_to_be_between':
                    col = df[kwargs['column']]
                    passed = (col >= kwargs['min_value']).all() and \
                             (col <= kwargs['max_value']).all()

                elif exp_type == 'expect_column_mean_to_be_between':
                    mean = df[kwargs['column']].mean()
                    passed = kwargs['min_value'] <= mean <= kwargs['max_value']

                else:
                    passed = False

                if passed:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['failures'].append({
                        'expectation': exp_type,
                        'column': kwargs.get('column'),
                        'details': kwargs
                    })

            except Exception as e:
                results['failed'] += 1
                results['failures'].append({
                    'expectation': exp_type,
                    'error': str(e)
                })

        # Print summary
        print(f"\nValidation Results:")
        print(f"  Total: {results['total']}")
        print(f"  Passed: {results['passed']}")
        print(f"  Failed: {results['failed']}")

        if results['failures']:
            print(f"\nFailures:")
            for failure in results['failures']:
                print(f"  - {failure}")

        return results


def main():
    """Validate generated data."""
    validator = DataValidator()

    results = validator.validate_interactions("data/raw/interactions.csv")

    # Save results
    with open("data/processed/validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    if results['failed'] > 0:
        print("\n⚠️  Validation failed!")
        return False
    else:
        print("\n✅ Validation passed!")
        return True


if __name__ == "__main__":
    main()
```

## Step 3: Feature Engineering

**File**: `src/data/features.py`

```python
"""
Feature engineering for feed ranking.

Generates user features, item features, and interaction features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class FeatureEngineer:
    """Engineer features for feed ranking."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.user_features_cache = {}
        self.item_features_cache = {}

    def engineer_features(
        self,
        interactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer all features.

        Args:
            interactions_df: Raw interaction data
            users_df: User metadata
            items_df: Item metadata

        Returns:
            DataFrame with all features
        """
        print("Engineering features...")

        # Convert timestamp
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])
        items_df['creation_date'] = pd.to_datetime(items_df['creation_date'])

        # Sort by time (important for temporal features)
        interactions_df = interactions_df.sort_values('timestamp').reset_index(drop=True)

        # Compute user features
        user_features = self._compute_user_features(interactions_df, users_df)

        # Compute item features
        item_features = self._compute_item_features(interactions_df, items_df)

        # Merge all features
        features_df = interactions_df.merge(user_features, on='user_id', how='left')
        features_df = features_df.merge(item_features, on='item_id', how='left')

        # Compute interaction features
        features_df = self._compute_interaction_features(features_df)

        # Select final feature columns
        feature_cols = [
            # Target
            'click',

            # IDs
            'user_id', 'item_id',

            # User features
            'user_historical_ctr', 'user_avg_dwell_time',
            'user_interaction_count', 'user_days_active',

            # Item features
            'item_ctr', 'item_avg_dwell_time',
            'item_popularity', 'item_age_days',

            # Interaction features
            'hour_of_day', 'day_of_week',
            'device_mobile', 'device_desktop', 'device_tablet',
            'user_item_previous_interactions',

            # Timestamp (for splitting)
            'timestamp'
        ]

        features_df = features_df[feature_cols].copy()

        # Handle missing values
        features_df = features_df.fillna(0)

        print(f"Engineered {len(features_df)} feature rows")

        return features_df

    def _compute_user_features(
        self,
        interactions_df: pd.DataFrame,
        users_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute user-level features."""
        print("  Computing user features...")

        # Aggregate historical behavior (use only clicks that happened before)
        user_stats = interactions_df.groupby('user_id').agg({
            'click': 'mean',  # CTR
            'dwell_time': 'mean',  # Average dwell time
            'interaction_id': 'count'  # Interaction count
        }).reset_index()

        user_stats.columns = [
            'user_id', 'user_historical_ctr',
            'user_avg_dwell_time', 'user_interaction_count'
        ]

        # Merge with user metadata
        user_features = users_df[['user_id', 'signup_date']].merge(
            user_stats, on='user_id', how='left'
        )

        # Days active (from signup to now)
        reference_date = interactions_df['timestamp'].max()
        user_features['user_days_active'] = (
            reference_date - user_features['signup_date']
        ).dt.days

        user_features = user_features.drop('signup_date', axis=1)

        # Fill NaN for users with no interactions
        user_features = user_features.fillna(0)

        return user_features

    def _compute_item_features(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute item-level features."""
        print("  Computing item features...")

        # Aggregate item stats
        item_stats = interactions_df.groupby('item_id').agg({
            'click': 'mean',  # CTR
            'dwell_time': 'mean',  # Average dwell time
            'interaction_id': 'count'  # Total impressions
        }).reset_index()

        item_stats.columns = [
            'item_id', 'item_ctr', 'item_avg_dwell_time', 'item_popularity'
        ]

        # Normalize popularity (log scale)
        item_stats['item_popularity'] = np.log1p(item_stats['item_popularity'])

        # Merge with item metadata
        item_features = items_df[['item_id', 'creation_date']].merge(
            item_stats, on='item_id', how='left'
        )

        # Item age
        reference_date = interactions_df['timestamp'].max()
        item_features['item_age_days'] = (
            reference_date - item_features['creation_date']
        ).dt.days

        item_features = item_features.drop('creation_date', axis=1)

        # Fill NaN for items with no interactions
        item_features = item_features.fillna(0)

        return item_features

    def _compute_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute interaction-level features."""
        print("  Computing interaction features...")

        # Time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Device one-hot encoding
        df['device_mobile'] = (df['device'] == 'mobile').astype(int)
        df['device_desktop'] = (df['device'] == 'desktop').astype(int)
        df['device_tablet'] = (df['device'] == 'tablet').astype(int)

        # User-item historical interaction count
        # (This is a simplified version; in production, use a feature store)
        user_item_counts = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
        df = df.merge(
            user_item_counts.rename(columns={'count': 'user_item_previous_interactions'}),
            on=['user_id', 'item_id'],
            how='left'
        )
        df['user_item_previous_interactions'] = df['user_item_previous_interactions'].fillna(0)

        # Subtract 1 to not count current interaction
        df['user_item_previous_interactions'] = (
            df['user_item_previous_interactions'] - 1
        ).clip(lower=0)

        return df

    def create_splits(
        self,
        features_df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> dict:
        """
        Create temporal train/val/test splits.

        Args:
            features_df: Engineered features
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing

        Returns:
            dict with 'train', 'val', 'test' DataFrames
        """
        print("\nCreating temporal splits...")

        # Sort by timestamp
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)

        # Calculate split indices
        n = len(features_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            'train': features_df.iloc[:train_end].copy(),
            'val': features_df.iloc[train_end:val_end].copy(),
            'test': features_df.iloc[val_end:].copy()
        }

        for split_name, split_df in splits.items():
            print(f"  {split_name}: {len(split_df)} rows "
                  f"({len(split_df)/n*100:.1f}%)")
            print(f"    Date range: {split_df['timestamp'].min()} to "
                  f"{split_df['timestamp'].max()}")
            print(f"    CTR: {split_df['click'].mean():.3f}")

        return splits


def main():
    """Run feature engineering pipeline."""
    # Load data
    print("Loading data...")
    interactions_df = pd.read_csv("data/raw/interactions.csv")
    users_df = pd.read_csv("data/raw/users.csv")
    items_df = pd.read_csv("data/raw/items.csv")

    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(interactions_df, users_df, items_df)

    # Create splits
    splits = engineer.create_splits(features_df)

    # Save splits
    print("\nSaving feature sets...")
    for split_name, split_df in splits.items():
        output_path = f"data/features/{split_name}_features.parquet"
        split_df.to_parquet(output_path, index=False)
        print(f"  Saved {output_path}")

    print("\n✅ Feature engineering complete!")


if __name__ == "__main__":
    main()
```

## Step 4: Build Airflow DAG

**File**: `dags/etl_dag.py`

```python
"""
ETL and Feature Engineering DAG.

Orchestrates:
1. Data generation (or ingestion)
2. Data validation
3. Feature engineering
4. Train/val/test splits
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/home/user/mlops-learning-plan/capstone_project')

from src.data.generator import FeedDataGenerator
from src.data.validator import DataValidator
from src.data.features import FeatureEngineer
import pandas as pd


default_args = {
    'owner': 'mlops_student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def generate_data():
    """Generate synthetic data."""
    print("Generating data...")
    generator = FeedDataGenerator()
    interactions_df = generator.generate_interactions()

    # Save
    interactions_df.to_csv("data/raw/interactions.csv", index=False)
    generator.users.to_csv("data/raw/users.csv", index=False)
    generator.items.to_csv("data/raw/items.csv", index=False)

    print("Data generation complete!")


def validate_data():
    """Validate data quality."""
    print("Validating data...")
    validator = DataValidator()
    results = validator.validate_interactions("data/raw/interactions.csv")

    if results['failed'] > 0:
        raise ValueError(f"Data validation failed: {results['failed']} expectations failed")

    print("Data validation passed!")


def engineer_features():
    """Engineer features."""
    print("Engineering features...")

    # Load data
    interactions_df = pd.read_csv("data/raw/interactions.csv")
    users_df = pd.read_csv("data/raw/users.csv")
    items_df = pd.read_csv("data/raw/items.csv")

    # Engineer
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(interactions_df, users_df, items_df)

    # Split
    splits = engineer.create_splits(features_df)

    # Save
    for split_name, split_df in splits.items():
        split_df.to_parquet(f"data/features/{split_name}_features.parquet", index=False)

    print("Feature engineering complete!")


with DAG(
    'etl_and_feature_engineering',
    default_args=default_args,
    description='ETL pipeline for feed ranking system',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 11, 1),
    catchup=False,
    tags=['etl', 'features'],
) as dag:

    # Task 1: Generate/ingest data
    generate_data_task = PythonOperator(
        task_id='generate_data',
        python_callable=generate_data,
    )

    # Task 2: Validate data
    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )

    # Task 3: Engineer features
    engineer_features_task = PythonOperator(
        task_id='engineer_features',
        python_callable=engineer_features,
    )

    # Task 4: Trigger training DAG (will implement in Module 3)
    # trigger_training = TriggerDagRunOperator(
    #     task_id='trigger_training',
    #     trigger_dag_id='model_training',
    # )

    # Dependencies
    generate_data_task >> validate_data_task >> engineer_features_task
    # >> trigger_training
```

## Testing

### Test the Pipeline

```bash
# Run components individually first
cd capstone_project

# 1. Test data generation
python src/data/generator.py

# 2. Test validation
python src/data/validator.py

# 3. Test feature engineering
python src/data/features.py

# 4. Test in Airflow
airflow dags test etl_and_feature_engineering 2024-11-15
```

### Verify Outputs

```bash
# Check data files
ls -lh data/raw/
ls -lh data/features/

# Inspect features
python -c "
import pandas as pd
df = pd.read_parquet('data/features/train_features.parquet')
print(df.head())
print(df.describe())
print(df['click'].mean())  # Should be CTR
"
```

## Review Checklist

- [ ] Data generator creates realistic interaction patterns
- [ ] Data validation catches quality issues
- [ ] Features are correctly engineered
- [ ] Train/val/test splits are temporal (not random)
- [ ] DAG runs successfully end-to-end
- [ ] Output files are in correct format (Parquet)
- [ ] Feature distributions are reasonable
- [ ] No data leakage between splits

## What to Submit

1. **Code Files**:
   - `src/data/generator.py`
   - `src/data/validator.py`
   - `src/data/features.py`
   - `dags/etl_dag.py`

2. **Sample Outputs**:
   - First 10 rows of `data/raw/interactions.csv`
   - Validation report
   - Feature summary statistics

3. **DAG Screenshot**: Airflow UI showing successful DAG run

4. **Reflection**:
   - How did you ensure data quality?
   - What feature engineering choices did you make and why?
   - What would you do differently at larger scale?

## Common Pitfalls

🚫 **Don't**:
- Use random train/test splits (breaks temporal dependency!)
- Engineer features using future information
- Ignore data validation
- Hardcode file paths

✅ **Do**:
- Use temporal splits
- Validate at every step
- Log feature distributions
- Make everything configurable

## Next Steps

Once your ETL pipeline is working:
1. Verify data quality thoroughly
2. Understand your features
3. Proceed to [Module 3: Model Training DAG](module3_training_pipeline.md)

---

**You've built the foundation!** Clean data and good features are 80% of ML success. Next, we'll train a ranking model on these features.
