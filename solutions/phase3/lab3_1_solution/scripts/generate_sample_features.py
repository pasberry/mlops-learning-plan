"""Generate sample CTR prediction features for training."""
import os
import numpy as np
import pandas as pd
from pathlib import Path


def generate_ctr_features(n_samples=10000, n_features=20):
    """Generate synthetic CTR prediction data.

    Features:
    - User features: age, gender, location (one-hot)
    - Ad features: category, size, position
    - Context features: hour, day_of_week, device_type
    - Interaction features: user-ad affinity scores

    Label: 1 if clicked, 0 if not
    """
    np.random.seed(42)

    # User features
    user_age = np.random.randint(18, 65, n_samples)
    user_gender = np.random.randint(0, 2, n_samples)  # 0: F, 1: M

    # Ad features
    ad_category = np.random.randint(0, 10, n_samples)  # 10 categories
    ad_size = np.random.choice(['small', 'medium', 'large'], n_samples)
    ad_position = np.random.randint(1, 6, n_samples)  # 1-5

    # Context features
    hour = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples)

    # Interaction features (synthetic affinity)
    user_ad_affinity = np.random.random(n_samples)
    recency_score = np.random.random(n_samples)

    # One-hot encode categorical features
    ad_size_small = (ad_size == 'small').astype(int)
    ad_size_medium = (ad_size == 'medium').astype(int)
    ad_size_large = (ad_size == 'large').astype(int)

    device_mobile = (device_type == 'mobile').astype(int)
    device_desktop = (device_type == 'desktop').astype(int)
    device_tablet = (device_type == 'tablet').astype(int)

    # Additional engineered features
    is_weekend = (day_of_week >= 5).astype(int)
    is_evening = ((hour >= 18) & (hour <= 22)).astype(int)

    # Create label (CTR) with some logic
    # Higher probability if:
    # - Good user-ad affinity
    # - Premium position
    # - Evening hours
    # - Mobile device
    click_prob = (
        0.1 +  # Base rate
        0.3 * user_ad_affinity +
        0.2 * (6 - ad_position) / 5 +  # Better positions
        0.15 * is_evening +
        0.1 * device_mobile +
        0.05 * (user_age > 30) +
        np.random.normal(0, 0.1, n_samples)  # Noise
    )
    click_prob = np.clip(click_prob, 0, 1)
    label = (np.random.random(n_samples) < click_prob).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'user_age': user_age,
        'user_gender': user_gender,
        'ad_category': ad_category,
        'ad_size_small': ad_size_small,
        'ad_size_medium': ad_size_medium,
        'ad_size_large': ad_size_large,
        'ad_position': ad_position,
        'hour': hour,
        'day_of_week': day_of_week,
        'device_mobile': device_mobile,
        'device_desktop': device_desktop,
        'device_tablet': device_tablet,
        'is_weekend': is_weekend,
        'is_evening': is_evening,
        'user_ad_affinity': user_ad_affinity,
        'recency_score': recency_score,
        'label': label
    })

    # Normalize numerical features
    numerical_cols = ['user_age', 'ad_position', 'hour', 'user_ad_affinity', 'recency_score']
    for col in numerical_cols:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data


def save_train_val_test_splits(data, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split and save data to parquet files."""
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Save
    for split, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_data.to_parquet(split_dir / 'data.parquet', index=False)
        print(f"Saved {len(split_data)} samples to {split_dir}/data.parquet")

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Val: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    print(f"  Positive rate (train): {train_data['label'].mean():.3f}")
    print(f"  Features: {len(data.columns) - 1}")


if __name__ == '__main__':
    # Generate features
    print("Generating sample CTR prediction features...")
    data = generate_ctr_features(n_samples=10000)

    # Save splits
    output_dir = 'data/features/v1'
    save_train_val_test_splits(data, output_dir)

    print(f"\n✅ Feature data ready at {output_dir}/")
    print("   Next: Build the PyTorch model!")
