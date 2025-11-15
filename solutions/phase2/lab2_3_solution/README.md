# Lab 2.3 Solution: Data Quality Validation

This solution provides a comprehensive data quality validation framework for e-commerce data using a custom DataValidator class and Airflow DAG.

## Overview

The data validation pipeline performs multiple types of quality checks:

1. **Schema Validation**: Column names and data types
2. **Completeness**: Missing values and null checks
3. **Uniqueness**: Duplicate detection
4. **Value Ranges**: Numeric bounds validation
5. **Categorical Values**: Allowed values for categorical fields
6. **Business Rules**: Custom business logic validation
7. **Referential Integrity**: Cross-table relationship checks
8. **Format Validation**: Email, date formats, etc.
9. **Statistical Properties**: Distribution checks

## Files

- `data_validator.py` - Complete DataValidator class with all validation methods
- `validation_dag.py` - Airflow DAG for automated validation
- `README.md` - This file

## Prerequisites

```bash
# Install required packages
pip install apache-airflow pandas numpy

# Ensure you have data to validate (from Lab 2.1)
# Expected files:
#   /tmp/ecommerce_data/raw/orders.csv
#   /tmp/ecommerce_data/raw/customers.csv
```

## Setup

### 1. Copy Files to Airflow

```bash
# Set AIRFLOW_HOME
export AIRFLOW_HOME=~/airflow

# Copy files to DAGs directory
cp validation_dag.py $AIRFLOW_HOME/dags/
cp data_validator.py $AIRFLOW_HOME/dags/
```

### 2. Start Airflow (if not already running)

**Terminal 1 - Webserver:**
```bash
airflow webserver --port 8080
```

**Terminal 2 - Scheduler:**
```bash
airflow scheduler
```

## Running Validation

### Option 1: Via Airflow UI

1. Go to http://localhost:8080
2. Find the DAG named `ecommerce_data_validation`
3. Toggle the DAG to "On"
4. Click the "Play" button to trigger a run
5. Monitor the execution - failed validations will be highlighted

### Option 2: Via Command Line

```bash
# Trigger the full validation pipeline
airflow dags trigger ecommerce_data_validation

# Test individual validation tasks
airflow tasks test ecommerce_data_validation validate_orders_schema 2024-01-01
airflow tasks test ecommerce_data_validation validate_orders_completeness 2024-01-01
airflow tasks test ecommerce_data_validation validate_orders_business_rules 2024-01-01
```

### Option 3: Use DataValidator Standalone

You can use the DataValidator class independently in your Python code:

```python
from data_validator import DataValidator
import pandas as pd

# Load your data
df = pd.read_csv('/tmp/ecommerce_data/raw/orders.csv')

# Create validator
validator = DataValidator()

# Run validations
validator.validate_schema(df, {
    'order_id': 'string',
    'quantity': 'int',
    'price': 'float'
})

validator.validate_missing_values(df, required_columns=['order_id', 'customer_id'])
validator.validate_value_ranges(df, {'quantity': (0, 1000), 'price': (0, 10000)})

# Get report
report = validator.get_validation_report()
print(f"Validation {'PASSED' if report['passed'] else 'FAILED'}")

# Save report
validator.save_report('/tmp/validation_report.json')
```

## Validation Checks Explained

### 1. Schema Validation

**Purpose**: Ensure data has correct structure

**Checks**:
- All expected columns are present
- No unexpected columns exist
- Data types match expectations

**Example**:
```python
expected_schema = {
    'order_id': 'string',
    'quantity': 'int',
    'price': 'float',
    'order_date': 'datetime'
}
validator.validate_schema(orders, expected_schema)
```

**Fails if**:
- Missing required columns
- Wrong data types (e.g., string instead of int)

### 2. Completeness Validation

**Purpose**: Detect missing values

**Checks**:
- Required columns have no null values
- Optional columns don't exceed missing value threshold

**Example**:
```python
validator.validate_missing_values(
    df,
    required_columns=['order_id', 'customer_id'],
    max_missing_pct=0.10  # Allow max 10% missing
)
```

**Fails if**:
- Required columns contain nulls
- Missing values exceed threshold

### 3. Uniqueness Validation

**Purpose**: Detect duplicates

**Checks**:
- Unique columns (like IDs) have no duplicates
- No fully duplicate rows

**Example**:
```python
validator.validate_duplicates(
    df,
    unique_columns=['order_id'],
    check_full_duplicates=True
)
```

**Fails if**:
- Duplicate IDs found
- Duplicate rows exist

### 4. Value Range Validation

**Purpose**: Ensure numeric values are within acceptable bounds

**Checks**:
- Values fall within min/max ranges
- No negative values where not allowed

**Example**:
```python
range_rules = {
    'quantity': (0, 1000),
    'price': (0, 10000),
    'age': (18, 100)
}
validator.validate_value_ranges(df, range_rules)
```

**Fails if**:
- Values below minimum
- Values above maximum

### 5. Categorical Values Validation

**Purpose**: Ensure categorical fields only contain allowed values

**Checks**:
- Category values are from predefined list
- No typos or invalid categories

**Example**:
```python
allowed_values = {
    'status': ['pending', 'processing', 'shipped', 'delivered'],
    'payment_method': ['Credit Card', 'PayPal', 'Debit Card']
}
validator.validate_categorical_values(df, allowed_values)
```

**Fails if**:
- Invalid category values found

### 6. Business Rules Validation

**Purpose**: Validate domain-specific business logic

**Checks**:
- Custom business constraints
- Calculated field relationships

**Example**:
```python
business_rules = [
    {
        'name': 'total_equals_price_times_quantity',
        'condition': 'total_amount == price * quantity',
        'description': 'Total must equal price × quantity'
    },
    {
        'name': 'positive_quantity',
        'condition': 'quantity > 0',
        'description': 'Quantity must be positive'
    }
]
validator.validate_business_rules(df, business_rules)
```

**Fails if**:
- Business rule violations found

### 7. Referential Integrity Validation

**Purpose**: Ensure relationships between tables are valid

**Checks**:
- Foreign keys exist in referenced table
- No orphaned records

**Example**:
```python
validator.validate_referential_integrity(
    orders,
    customers,
    'customer_id',
    df1_name='orders',
    df2_name='customers'
)
```

**Fails if**:
- Orders reference non-existent customers

### 8. Format Validation

**Purpose**: Validate specific formats (email, phone, etc.)

**Checks**:
- Email addresses match pattern
- Dates are in correct format

**Example**:
```python
validator.validate_email_format(customers, 'email')
validator.validate_date_formats(orders, ['order_date'], '%Y-%m-%d')
```

**Fails if**:
- Invalid email format
- Dates don't match pattern
- Future dates where not allowed

### 9. Statistical Properties Validation

**Purpose**: Detect data distribution anomalies

**Checks**:
- Mean and standard deviation within expected ranges
- Distribution hasn't changed significantly

**Example**:
```python
stats_rules = {
    'price': {
        'mean': 100,
        'std': 50,
        'tolerance': 0.2  # 20% tolerance
    }
}
validator.validate_statistical_properties(df, stats_rules)
```

**Fails if**:
- Statistics deviate beyond tolerance

## Output Structure

```
/tmp/ecommerce_data/validation/
├── orders_schema_validation.json
├── orders_completeness_validation.json
├── orders_uniqueness_validation.json
├── orders_ranges_validation.json
├── orders_categorical_validation.json
├── orders_business_rules_validation.json
├── customers_schema_validation.json
├── customers_quality_validation.json
├── referential_integrity_validation.json
└── comprehensive_validation_report.json
```

## Understanding Validation Reports

### Individual Validation Report

```json
{
  "check": "value_ranges",
  "passed": false,
  "issues": [
    {
      "type": "values_below_minimum",
      "column": "quantity",
      "min_allowed": 0,
      "min_found": -5,
      "count": 12
    }
  ]
}
```

### Comprehensive Report

```json
{
  "timestamp": "2024-01-01 12:00:00",
  "overall_status": "FAILED",
  "validations": {
    "orders": {
      "schema": true,
      "completeness": true,
      "uniqueness": true,
      "ranges": false,
      "categorical": true,
      "business_rules": false
    },
    "customers": {
      "schema": true,
      "quality": false
    }
  },
  "summary": {
    "total_checks": 9,
    "passed_checks": 6,
    "failed_checks": 3
  }
}
```

## Common Validation Issues and Fixes

### Issue 1: Missing Values in Required Columns

**Problem**:
```
Missing values found in required column 'customer_id'
```

**Fix**:
```python
# Remove rows with missing customer_id
df = df.dropna(subset=['customer_id'])

# Or impute with placeholder
df['customer_id'].fillna('UNKNOWN', inplace=True)
```

### Issue 2: Negative Values

**Problem**:
```
12 records have negative quantity values
```

**Fix**:
```python
# Convert to absolute value
df.loc[df['quantity'] < 0, 'quantity'] = df['quantity'].abs()

# Or filter out
df = df[df['quantity'] >= 0]
```

### Issue 3: Invalid Email Format

**Problem**:
```
25 customers have invalid email addresses
```

**Fix**:
```python
# Fix missing domain
invalid_mask = ~df['email'].str.contains('@', na=False)
df.loc[invalid_mask, 'email'] = df.loc[invalid_mask, 'customer_id'] + '@placeholder.com'
```

### Issue 4: Duplicate Order IDs

**Problem**:
```
5 duplicate order_id values found
```

**Fix**:
```python
# Keep first occurrence
df = df.drop_duplicates(subset=['order_id'], keep='first')

# Or generate new IDs for duplicates
df.loc[df.duplicated('order_id'), 'order_id'] = df['order_id'] + '_' + df.index.astype(str)
```

### Issue 5: Referential Integrity Violation

**Problem**:
```
100 orders reference customer_ids that don't exist
```

**Fix**:
```python
# Remove orders with invalid customer_ids
valid_customer_ids = customers['customer_id'].unique()
df = df[df['customer_id'].isin(valid_customer_ids)]

# Or create placeholder customers
missing_ids = df[~df['customer_id'].isin(valid_customer_ids)]['customer_id'].unique()
for cust_id in missing_ids:
    customers = customers.append({'customer_id': cust_id, 'email': f'{cust_id}@unknown.com'})
```

## Integration with ETL Pipeline

Integrate validation into your ETL pipeline:

```python
# In your ETL DAG
def validate_and_clean(**context):
    # Load raw data
    orders = pd.read_csv('raw_orders.csv')

    # Validate
    validator = DataValidator()
    validator.validate_schema(orders, expected_schema)
    validator.validate_business_rules(orders, business_rules)

    report = validator.get_validation_report()

    if not report['passed']:
        # Log issues
        logger.warning(f"Validation issues found: {report}")

        # Attempt to fix common issues
        orders = fix_data_quality_issues(orders)

        # Re-validate
        validator2 = DataValidator()
        validator2.validate_schema(orders, expected_schema)
        report2 = validator2.get_validation_report()

        if not report2['passed']:
            raise ValueError("Data quality issues could not be fixed")

    # Continue with clean data
    orders.to_csv('cleaned_orders.csv', index=False)
```

## Monitoring and Alerting

### Set up Email Alerts

In `validation_dag.py`, configure email alerts:

```python
default_args = {
    'email': ['data-team@example.com'],
    'email_on_failure': True,  # Alert when validation fails
}
```

### Create Dashboard

Track validation metrics over time:

```python
import matplotlib.pyplot as plt

# Load historical validation reports
reports = []
for file in glob.glob('/tmp/ecommerce_data/validation/comprehensive_*.json'):
    with open(file) as f:
        reports.append(json.load(f))

# Plot pass rate over time
dates = [r['timestamp'] for r in reports]
pass_rates = [r['summary']['passed_checks'] / r['summary']['total_checks'] for r in reports]

plt.plot(dates, pass_rates)
plt.title('Data Quality Over Time')
plt.ylabel('Pass Rate')
plt.xlabel('Date')
plt.savefig('quality_trend.png')
```

## Best Practices

1. **Validate Early**: Run validation immediately after data ingestion
2. **Fail Fast**: Stop pipeline if critical validations fail
3. **Log Everything**: Keep detailed validation logs
4. **Version Validation Rules**: Track changes to validation logic
5. **Monitor Trends**: Track quality metrics over time
6. **Automate Fixes**: Create automated fixes for common issues
7. **Alert on Failures**: Set up notifications for validation failures
8. **Document Rules**: Clearly document what each validation checks

## Advanced Usage

### Custom Validators

Add custom validation methods:

```python
class CustomValidator(DataValidator):
    def validate_product_sku_format(self, df, sku_column='sku'):
        """Validate SKU format: ABC-1234"""
        pattern = r'^[A-Z]{3}-\d{4}$'
        invalid = ~df[sku_column].str.match(pattern, na=False)

        if invalid.any():
            self.validation_results['checks'].append({
                'check': 'sku_format',
                'passed': False,
                'issues': [{
                    'type': 'invalid_sku_format',
                    'count': int(invalid.sum())
                }]
            })
```

### Data Quality Scoring

Calculate overall data quality score:

```python
def calculate_quality_score(validation_report):
    """Calculate 0-100 quality score"""
    total = validation_report['summary']['total_checks']
    passed = validation_report['summary']['passed_checks']
    return (passed / total) * 100

score = calculate_quality_score(report)
print(f"Data Quality Score: {score}/100")
```

## Troubleshooting

### Validation DAG Not Running

```bash
# Check DAG is enabled
airflow dags list | grep validation

# Check for errors
airflow dags list-import-errors

# Test DAG file
python $AIRFLOW_HOME/dags/validation_dag.py
```

### Import Errors

```bash
# Ensure data_validator.py is in DAGs directory
ls $AIRFLOW_HOME/dags/data_validator.py

# Check Python path
export PYTHONPATH=$PYTHONPATH:$AIRFLOW_HOME/dags
```

### No Data to Validate

```bash
# Check if data exists
ls -l /tmp/ecommerce_data/raw/

# Run ETL pipeline first
airflow dags trigger ecommerce_etl_pipeline
```

## Next Steps

1. **Lab 2.4**: Add scheduling to run validation daily
2. **Production**: Integrate with data quality monitoring tools
3. **Expand**: Add more validation rules as data evolves
4. **Optimize**: Profile validation performance for large datasets
5. **Integrate**: Connect to data catalog and lineage tools

## Resources

- [Great Expectations](https://greatexpectations.io/) - Alternative validation framework
- [Deequ](https://github.com/awslabs/deequ) - AWS data quality library
- [Data Quality Best Practices](https://cloud.google.com/architecture/dq-methodology)

## License

MIT License - Free to use for learning purposes
