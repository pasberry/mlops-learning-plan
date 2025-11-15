# Lab 2.1 Solution: ETL Pipeline with Apache Airflow

This solution provides a complete ETL (Extract, Transform, Load) pipeline for processing e-commerce data using Apache Airflow.

## Overview

The ETL pipeline consists of the following stages:

1. **Ingest**: Load raw data from source
2. **Validate**: Check data quality and business rules
3. **Clean**: Handle missing values, outliers, and duplicates
4. **Transform**: Create aggregated views and analytics datasets
5. **Report**: Generate comprehensive pipeline execution report

## Files

- `generate_ecommerce_data.py` - Synthetic e-commerce data generator
- `etl_pipeline_dag.py` - Complete Airflow DAG implementation
- `README.md` - This file

## Prerequisites

```bash
# Install required packages
pip install apache-airflow pandas numpy

# Initialize Airflow database (first time only)
airflow db init

# Create an Airflow user (first time only)
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

## Setup

### 1. Configure Airflow

```bash
# Set AIRFLOW_HOME (optional, defaults to ~/airflow)
export AIRFLOW_HOME=~/airflow

# Create DAGs directory
mkdir -p $AIRFLOW_HOME/dags

# Copy the DAG file
cp etl_pipeline_dag.py $AIRFLOW_HOME/dags/
cp generate_ecommerce_data.py $AIRFLOW_HOME/dags/
```

### 2. Start Airflow

Open two terminal windows:

**Terminal 1 - Webserver:**
```bash
airflow webserver --port 8080
```

**Terminal 2 - Scheduler:**
```bash
airflow scheduler
```

### 3. Access Airflow UI

Open your browser and navigate to: http://localhost:8080

- Username: `admin`
- Password: `admin` (or what you set during user creation)

## Running the Pipeline

### Option 1: Via Airflow UI

1. Go to http://localhost:8080
2. Find the DAG named `ecommerce_etl_pipeline`
3. Toggle the DAG to "On" (enable)
4. Click the "Play" button to trigger a manual run
5. Monitor the execution in the Graph or Grid view

### Option 2: Via Command Line

```bash
# Test individual tasks
airflow tasks test ecommerce_etl_pipeline create_directories 2024-01-01
airflow tasks test ecommerce_etl_pipeline ingest_data 2024-01-01
airflow tasks test ecommerce_etl_pipeline validate_data 2024-01-01
airflow tasks test ecommerce_etl_pipeline clean_data 2024-01-01
airflow tasks test ecommerce_etl_pipeline transform_data 2024-01-01
airflow tasks test ecommerce_etl_pipeline generate_report 2024-01-01

# Trigger a full DAG run
airflow dags trigger ecommerce_etl_pipeline

# Check DAG status
airflow dags list
airflow dags state ecommerce_etl_pipeline
```

### Option 3: Standalone Data Generation

You can also run the data generator independently:

```bash
# Generate sample data
python generate_ecommerce_data.py
```

This will create sample data in `/tmp/ecommerce_data/`.

## Pipeline Stages Explained

### Stage 1: Ingest Data

```python
Task: ingest_data
Purpose: Load raw data from source
Output:
  - /tmp/ecommerce_data/raw/orders.csv
  - /tmp/ecommerce_data/raw/customers.csv
```

Generates synthetic e-commerce data including:
- Orders (order_id, customer_id, product details, amounts, location)
- Customers (customer_id, email, registration date, demographics)

### Stage 2: Validate Data

```python
Task: validate_data
Purpose: Check data quality and identify issues
Output:
  - /tmp/ecommerce_data/validated/validation_report.json
  - Validated CSV files
```

Validation checks:
- Missing values
- Negative quantities/amounts
- Invalid emails
- Age outliers
- Duplicate IDs

### Stage 3: Clean Data

```python
Task: clean_data
Purpose: Fix data quality issues
Output:
  - /tmp/ecommerce_data/cleaned/orders.csv
  - /tmp/ecommerce_data/cleaned/customers.csv
  - /tmp/ecommerce_data/cleaned/cleaning_report.json
```

Cleaning operations:
- Remove duplicates
- Handle missing values
- Fix negative values
- Correct invalid emails
- Clip age outliers
- Standardize date formats

### Stage 4: Transform Data

```python
Task: transform_data
Purpose: Create analytics-ready datasets
Output:
  - customer_summary.csv - Customer metrics (orders, spending)
  - category_performance.csv - Product category analysis
  - daily_sales.csv - Daily sales trends
  - country_performance.csv - Geographic performance
  - enhanced_customers.csv - Enriched customer data
```

Transformations include:
- Customer lifetime value calculations
- Product category performance metrics
- Time-based aggregations
- Geographic analysis

### Stage 5: Generate Report

```python
Task: generate_report
Purpose: Create comprehensive pipeline report
Output:
  - /tmp/ecommerce_data/transformed/etl_pipeline_report.json
```

Report includes:
- Pipeline execution metadata
- Data quality metrics
- Cleaning statistics
- Transformation results

## Output Directory Structure

```
/tmp/ecommerce_data/
├── raw/
│   ├── orders.csv
│   └── customers.csv
├── validated/
│   ├── orders.csv
│   ├── customers.csv
│   └── validation_report.json
├── cleaned/
│   ├── orders.csv
│   ├── customers.csv
│   └── cleaning_report.json
└── transformed/
    ├── customer_summary.csv
    ├── category_performance.csv
    ├── daily_sales.csv
    ├── country_performance.csv
    ├── enhanced_customers.csv
    ├── transformation_report.json
    └── etl_pipeline_report.json
```

## Monitoring and Debugging

### Check Logs

```bash
# View task logs
airflow tasks log ecommerce_etl_pipeline ingest_data 2024-01-01

# View DAG processor logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/
```

### Check XCom Data

XCom (cross-communication) is used to pass data between tasks:

```bash
# Access via UI: Admin > XCom
# Or query via command line:
airflow tasks xcom-list ecommerce_etl_pipeline
```

### Common Issues

1. **DAG not appearing in UI**
   - Check DAG file syntax: `python etl_pipeline_dag.py`
   - Verify file is in `$AIRFLOW_HOME/dags/`
   - Check scheduler logs

2. **Import errors**
   - Ensure `generate_ecommerce_data.py` is in the same directory as DAG
   - Or add to Python path

3. **Permission errors**
   - Ensure `/tmp/ecommerce_data/` is writable
   - Check Airflow user permissions

## Data Quality Issues (Intentional)

The generator creates intentional data quality issues for testing:

- **5% missing values** - Null customer_ids, amounts
- **2% negative values** - Negative quantities/amounts
- **5% invalid emails** - Missing domain or @ symbol
- **5% age outliers** - Ages < 18 or > 100

These issues are detected in validation and fixed in cleaning.

## Customization

### Adjust Data Volume

Edit `generate_ecommerce_data.py`:

```python
# Generate more/less data
orders = generator.generate_orders(num_orders=5000)  # Default: 1000
customers = generator.generate_customer_data(num_customers=1000)  # Default: 500
```

### Change Schedule

Edit `etl_pipeline_dag.py`:

```python
# Run daily at 2 AM
schedule_interval='0 2 * * *'

# Run hourly
schedule_interval='@hourly'

# Run weekly
schedule_interval='@weekly'
```

### Add Custom Transformations

Add new transformation functions in `transform_data()`:

```python
def transform_data(**context):
    # ... existing code ...

    # Add custom transformation
    logger.info("Creating custom transformation...")
    custom_df = orders.groupby(['country', 'product_category']).agg({
        'total_amount': 'sum'
    })
    custom_df.to_csv(f'{TRANSFORMED_DATA_DIR}/custom_analysis.csv')
```

## Testing

### Unit Test Individual Functions

```python
# Test data generation
python -c "from generate_ecommerce_data import EcommerceDataGenerator; \
    g = EcommerceDataGenerator(); \
    orders = g.generate_orders(100); \
    print(f'Generated {len(orders)} orders')"

# Test validation logic
python -c "from etl_pipeline_dag import validate_data; \
    import pandas as pd; \
    # ... test validation ..."
```

### Integration Test

Run the full pipeline with test data:

```bash
# Trigger test run
airflow dags test ecommerce_etl_pipeline 2024-01-01
```

## Performance Optimization

For large datasets:

1. **Use chunking for file processing**:
   ```python
   for chunk in pd.read_csv('large_file.csv', chunksize=10000):
       process(chunk)
   ```

2. **Enable parallel execution** in `airflow.cfg`:
   ```ini
   [core]
   parallelism = 32
   dag_concurrency = 16
   ```

3. **Use appropriate executor**:
   - LocalExecutor: Single machine, parallel tasks
   - CeleryExecutor: Distributed execution
   - KubernetesExecutor: Kubernetes-based scaling

## Next Steps

1. **Lab 2.2**: Add feature engineering to this pipeline
2. **Lab 2.3**: Implement comprehensive data quality checks
3. **Lab 2.4**: Add scheduling and backfill capabilities
4. **Production**:
   - Add data lineage tracking
   - Implement alerting
   - Set up monitoring dashboards
   - Add data versioning

## Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## License

MIT License - Free to use for learning purposes
