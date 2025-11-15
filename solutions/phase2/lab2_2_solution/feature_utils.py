"""
Feature Engineering Utilities
Functions for creating features from e-commerce data including RFM, temporal, and behavioral features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rfm_features(orders_df, customer_id_col='customer_id',
                          order_date_col='order_date',
                          amount_col='total_amount',
                          reference_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) features for customers.

    Args:
        orders_df: DataFrame with order data
        customer_id_col: Name of customer ID column
        order_date_col: Name of order date column
        amount_col: Name of amount column
        reference_date: Reference date for recency calculation (default: max order date)

    Returns:
        DataFrame with RFM features per customer
    """
    logger.info("Calculating RFM features...")

    # Ensure order_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(orders_df[order_date_col]):
        orders_df[order_date_col] = pd.to_datetime(orders_df[order_date_col])

    # Set reference date
    if reference_date is None:
        reference_date = orders_df[order_date_col].max()
    elif isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)

    # Calculate RFM metrics
    rfm = orders_df.groupby(customer_id_col).agg({
        order_date_col: lambda x: (reference_date - x.max()).days,  # Recency
        amount_col: ['count', 'sum', 'mean', 'std', 'min', 'max']   # Frequency & Monetary
    }).reset_index()

    # Flatten column names
    rfm.columns = [customer_id_col, 'recency_days', 'frequency', 'monetary_total',
                   'monetary_mean', 'monetary_std', 'monetary_min', 'monetary_max']

    # Fill NaN std with 0 (for customers with single order)
    rfm['monetary_std'] = rfm['monetary_std'].fillna(0)

    # Calculate RFM scores (1-5 scale)
    rfm['recency_score'] = pd.qcut(rfm['recency_days'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm['frequency_score'] = pd.qcut(rfm['frequency'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['monetary_score'] = pd.qcut(rfm['monetary_total'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

    # Convert scores to int
    rfm['recency_score'] = rfm['recency_score'].astype(int)
    rfm['frequency_score'] = rfm['frequency_score'].astype(int)
    rfm['monetary_score'] = rfm['monetary_score'].astype(int)

    # Calculate combined RFM score
    rfm['rfm_score'] = (rfm['recency_score'] + rfm['frequency_score'] + rfm['monetary_score']) / 3

    # Segment customers based on RFM score
    rfm['customer_segment'] = pd.cut(rfm['rfm_score'],
                                     bins=[0, 2, 3, 4, 5],
                                     labels=['At Risk', 'Needs Attention', 'Loyal', 'Champions'])

    logger.info(f"RFM features calculated for {len(rfm)} customers")
    return rfm


def calculate_temporal_features(orders_df, customer_id_col='customer_id',
                                order_date_col='order_date'):
    """
    Calculate temporal features from order history.

    Args:
        orders_df: DataFrame with order data
        customer_id_col: Name of customer ID column
        order_date_col: Name of order date column

    Returns:
        DataFrame with temporal features per customer
    """
    logger.info("Calculating temporal features...")

    # Ensure order_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(orders_df[order_date_col]):
        orders_df[order_date_col] = pd.to_datetime(orders_df[order_date_col])

    temporal_features = []

    for customer_id, group in orders_df.groupby(customer_id_col):
        # Sort by date
        group = group.sort_values(order_date_col)

        # Basic temporal features
        first_order = group[order_date_col].min()
        last_order = group[order_date_col].max()
        customer_lifetime_days = (last_order - first_order).days

        # Calculate inter-purchase intervals
        if len(group) > 1:
            intervals = group[order_date_col].diff().dt.days.dropna()
            avg_days_between_orders = intervals.mean()
            std_days_between_orders = intervals.std()
            min_days_between_orders = intervals.min()
            max_days_between_orders = intervals.max()
        else:
            avg_days_between_orders = 0
            std_days_between_orders = 0
            min_days_between_orders = 0
            max_days_between_orders = 0

        # Day of week patterns
        group['day_of_week'] = group[order_date_col].dt.dayofweek
        group['hour_of_day'] = group[order_date_col].dt.hour
        most_common_day = group['day_of_week'].mode()[0] if len(group['day_of_week'].mode()) > 0 else 0
        most_common_hour = group['hour_of_day'].mode()[0] if len(group['hour_of_day'].mode()) > 0 else 0

        # Weekend vs weekday orders
        weekend_orders = (group['day_of_week'] >= 5).sum()
        weekday_orders = (group['day_of_week'] < 5).sum()
        weekend_ratio = weekend_orders / len(group) if len(group) > 0 else 0

        temporal_features.append({
            customer_id_col: customer_id,
            'first_order_date': first_order,
            'last_order_date': last_order,
            'customer_lifetime_days': customer_lifetime_days,
            'avg_days_between_orders': avg_days_between_orders,
            'std_days_between_orders': std_days_between_orders,
            'min_days_between_orders': min_days_between_orders,
            'max_days_between_orders': max_days_between_orders,
            'most_common_order_day': most_common_day,
            'most_common_order_hour': most_common_hour,
            'weekend_order_ratio': weekend_ratio,
            'weekday_orders': weekday_orders,
            'weekend_orders': weekend_orders
        })

    temporal_df = pd.DataFrame(temporal_features)
    logger.info(f"Temporal features calculated for {len(temporal_df)} customers")
    return temporal_df


def calculate_behavioral_features(orders_df, customer_id_col='customer_id',
                                  product_category_col='product_category',
                                  quantity_col='quantity',
                                  status_col='status'):
    """
    Calculate behavioral features based on purchase patterns.

    Args:
        orders_df: DataFrame with order data
        customer_id_col: Name of customer ID column
        product_category_col: Name of product category column
        quantity_col: Name of quantity column
        status_col: Name of order status column

    Returns:
        DataFrame with behavioral features per customer
    """
    logger.info("Calculating behavioral features...")

    behavioral_features = []

    for customer_id, group in orders_df.groupby(customer_id_col):
        # Category preferences
        category_counts = group[product_category_col].value_counts()
        most_purchased_category = category_counts.index[0] if len(category_counts) > 0 else 'Unknown'
        num_unique_categories = group[product_category_col].nunique()
        category_diversity = num_unique_categories / len(group) if len(group) > 0 else 0

        # Quantity patterns
        avg_items_per_order = group[quantity_col].mean()
        total_items_purchased = group[quantity_col].sum()
        max_items_in_order = group[quantity_col].max()

        # Order status patterns
        if status_col in group.columns:
            cancelled_orders = (group[status_col] == 'cancelled').sum()
            delivered_orders = (group[status_col] == 'delivered').sum()
            cancellation_rate = cancelled_orders / len(group) if len(group) > 0 else 0
            delivery_rate = delivered_orders / len(group) if len(group) > 0 else 0
        else:
            cancelled_orders = 0
            delivered_orders = len(group)
            cancellation_rate = 0
            delivery_rate = 1.0

        # Bulk buyer indicator
        is_bulk_buyer = (group[quantity_col] > 3).sum() / len(group) if len(group) > 0 else 0

        behavioral_features.append({
            customer_id_col: customer_id,
            'most_purchased_category': most_purchased_category,
            'num_unique_categories': num_unique_categories,
            'category_diversity_score': category_diversity,
            'avg_items_per_order': avg_items_per_order,
            'total_items_purchased': int(total_items_purchased),
            'max_items_in_order': int(max_items_in_order),
            'cancelled_orders': int(cancelled_orders),
            'delivered_orders': int(delivered_orders),
            'cancellation_rate': cancellation_rate,
            'delivery_rate': delivery_rate,
            'bulk_buyer_score': is_bulk_buyer
        })

    behavioral_df = pd.DataFrame(behavioral_features)
    logger.info(f"Behavioral features calculated for {len(behavioral_df)} customers")
    return behavioral_df


def calculate_customer_value_features(orders_df, customers_df=None,
                                      customer_id_col='customer_id',
                                      order_date_col='order_date',
                                      amount_col='total_amount'):
    """
    Calculate customer value and engagement features.

    Args:
        orders_df: DataFrame with order data
        customers_df: Optional DataFrame with customer data
        customer_id_col: Name of customer ID column
        order_date_col: Name of order date column
        amount_col: Name of amount column

    Returns:
        DataFrame with customer value features
    """
    logger.info("Calculating customer value features...")

    # Ensure order_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(orders_df[order_date_col]):
        orders_df[order_date_col] = pd.to_datetime(orders_df[order_date_col])

    value_features = []

    for customer_id, group in orders_df.groupby(customer_id_col):
        # Sort by date
        group = group.sort_values(order_date_col)

        # Customer Lifetime Value (CLV)
        total_revenue = group[amount_col].sum()
        avg_order_value = group[amount_col].mean()

        # Customer tenure
        first_order = group[order_date_col].min()
        last_order = group[order_date_col].max()
        tenure_days = (datetime.now() - first_order).days

        # Purchase frequency
        num_orders = len(group)
        if tenure_days > 0:
            orders_per_month = (num_orders / tenure_days) * 30
        else:
            orders_per_month = 0

        # Trend analysis (recent vs historical)
        if len(group) >= 6:
            recent_6_orders = group.tail(6)[amount_col].mean()
            historical_orders = group.head(len(group) - 6)[amount_col].mean()
            value_trend = (recent_6_orders - historical_orders) / historical_orders if historical_orders > 0 else 0
        else:
            value_trend = 0

        # Engagement score (combination of frequency and recency)
        days_since_last_order = (datetime.now() - last_order).days
        if days_since_last_order == 0:
            days_since_last_order = 1  # Avoid division by zero
        engagement_score = (num_orders / days_since_last_order) * 100

        # Predict next purchase (simple estimate based on avg interval)
        if len(group) > 1:
            intervals = group[order_date_col].diff().dt.days.dropna()
            avg_interval = intervals.mean()
            predicted_next_purchase_days = avg_interval
        else:
            predicted_next_purchase_days = None

        value_features.append({
            customer_id_col: customer_id,
            'total_revenue': total_revenue,
            'avg_order_value': avg_order_value,
            'tenure_days': tenure_days,
            'num_orders': num_orders,
            'orders_per_month': orders_per_month,
            'value_trend': value_trend,
            'engagement_score': engagement_score,
            'days_since_last_order': days_since_last_order,
            'predicted_next_purchase_days': predicted_next_purchase_days
        })

    value_df = pd.DataFrame(value_features)

    # If customer data is provided, add customer-specific features
    if customers_df is not None:
        if 'registration_date' in customers_df.columns:
            customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])
            customers_df['account_age_days'] = (datetime.now() - customers_df['registration_date']).dt.days

        value_df = value_df.merge(
            customers_df[[customer_id_col, 'account_age_days', 'is_premium']]
            if 'account_age_days' in customers_df.columns
            else customers_df[[customer_id_col]],
            on=customer_id_col,
            how='left'
        )

    logger.info(f"Customer value features calculated for {len(value_df)} customers")
    return value_df


def create_aggregated_features(orders_df, customer_id_col='customer_id',
                               time_window_days=30):
    """
    Create time-windowed aggregated features.

    Args:
        orders_df: DataFrame with order data
        customer_id_col: Name of customer ID column
        time_window_days: Number of days for time window (default: 30)

    Returns:
        DataFrame with aggregated features per customer
    """
    logger.info(f"Creating aggregated features with {time_window_days}-day window...")

    # Ensure order_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(orders_df['order_date']):
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

    # Define cutoff date
    cutoff_date = datetime.now() - timedelta(days=time_window_days)

    # Filter to recent orders
    recent_orders = orders_df[orders_df['order_date'] >= cutoff_date]

    # Aggregate features
    agg_features = recent_orders.groupby(customer_id_col).agg({
        'order_id': 'count',
        'total_amount': ['sum', 'mean', 'max'],
        'quantity': 'sum',
        'product_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).reset_index()

    # Flatten column names
    agg_features.columns = [
        customer_id_col,
        f'orders_last_{time_window_days}d',
        f'revenue_last_{time_window_days}d',
        f'avg_order_value_last_{time_window_days}d',
        f'max_order_value_last_{time_window_days}d',
        f'items_purchased_last_{time_window_days}d',
        f'top_category_last_{time_window_days}d'
    ]

    logger.info(f"Aggregated features created for {len(agg_features)} active customers")
    return agg_features


def combine_all_features(orders_df, customers_df=None, customer_id_col='customer_id'):
    """
    Combine all feature engineering functions into a comprehensive feature set.

    Args:
        orders_df: DataFrame with order data
        customers_df: Optional DataFrame with customer data
        customer_id_col: Name of customer ID column

    Returns:
        DataFrame with all engineered features
    """
    logger.info("Combining all features...")

    # Calculate all feature sets
    rfm_features = calculate_rfm_features(orders_df, customer_id_col=customer_id_col)
    temporal_features = calculate_temporal_features(orders_df, customer_id_col=customer_id_col)
    behavioral_features = calculate_behavioral_features(orders_df, customer_id_col=customer_id_col)
    value_features = calculate_customer_value_features(orders_df, customers_df, customer_id_col=customer_id_col)
    agg_features_30d = create_aggregated_features(orders_df, customer_id_col=customer_id_col, time_window_days=30)
    agg_features_90d = create_aggregated_features(orders_df, customer_id_col=customer_id_col, time_window_days=90)

    # Merge all features
    combined = rfm_features
    combined = combined.merge(temporal_features, on=customer_id_col, how='left')
    combined = combined.merge(behavioral_features, on=customer_id_col, how='left')
    combined = combined.merge(value_features, on=customer_id_col, how='left')
    combined = combined.merge(agg_features_30d, on=customer_id_col, how='left')
    combined = combined.merge(agg_features_90d, on=customer_id_col, how='left', suffixes=('', '_90d'))

    # Fill NaN values for customers without recent activity
    for col in combined.columns:
        if combined[col].dtype in [np.float64, np.int64]:
            combined[col] = combined[col].fillna(0)

    logger.info(f"Combined feature set created with {len(combined.columns)} features for {len(combined)} customers")
    return combined


def create_feature_summary(features_df):
    """
    Create a summary report of engineered features.

    Args:
        features_df: DataFrame with engineered features

    Returns:
        Dictionary with feature summary statistics
    """
    summary = {
        'total_customers': len(features_df),
        'total_features': len(features_df.columns),
        'feature_categories': {
            'rfm': len([col for col in features_df.columns if 'rfm' in col.lower() or
                       'recency' in col.lower() or 'frequency' in col.lower() or
                       'monetary' in col.lower()]),
            'temporal': len([col for col in features_df.columns if 'days' in col.lower() or
                           'date' in col.lower() or 'time' in col.lower()]),
            'behavioral': len([col for col in features_df.columns if 'category' in col.lower() or
                             'items' in col.lower() or 'cancel' in col.lower()]),
            'value': len([col for col in features_df.columns if 'revenue' in col.lower() or
                         'value' in col.lower() or 'clv' in col.lower()])
        },
        'segment_distribution': features_df['customer_segment'].value_counts().to_dict() if 'customer_segment' in features_df.columns else {},
        'numeric_feature_stats': features_df.describe().to_dict()
    }

    return summary
