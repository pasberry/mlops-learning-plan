"""
Data Validator
Comprehensive data quality validation for e-commerce data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validator for e-commerce data.
    Provides schema validation, business rule checks, and data quality metrics.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data validator.

        Args:
            config: Optional configuration dictionary with validation rules
        """
        self.config = config or {}
        self.validation_results = {
            'timestamp': str(datetime.now()),
            'passed': True,
            'checks': []
        }

    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, str]) -> Dict:
        """
        Validate DataFrame schema against expected schema.

        Args:
            df: DataFrame to validate
            expected_schema: Dictionary mapping column names to expected types

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating schema...")

        results = {
            'check': 'schema_validation',
            'passed': True,
            'issues': []
        }

        # Check for missing columns
        expected_columns = set(expected_schema.keys())
        actual_columns = set(df.columns)
        missing_columns = expected_columns - actual_columns

        if missing_columns:
            results['passed'] = False
            results['issues'].append({
                'type': 'missing_columns',
                'columns': list(missing_columns)
            })

        # Check for extra columns
        extra_columns = actual_columns - expected_columns
        if extra_columns:
            results['issues'].append({
                'type': 'extra_columns',
                'columns': list(extra_columns)
            })

        # Check data types
        type_mismatches = []
        for col, expected_type in expected_schema.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._types_compatible(actual_type, expected_type):
                    type_mismatches.append({
                        'column': col,
                        'expected': expected_type,
                        'actual': actual_type
                    })

        if type_mismatches:
            results['passed'] = False
            results['issues'].append({
                'type': 'type_mismatches',
                'mismatches': type_mismatches
            })

        logger.info(f"Schema validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_missing_values(self, df: pd.DataFrame,
                               required_columns: Optional[List[str]] = None,
                               max_missing_pct: float = 0.05) -> Dict:
        """
        Validate missing values in DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: Columns that cannot have missing values
            max_missing_pct: Maximum allowed percentage of missing values (0-1)

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating missing values...")

        results = {
            'check': 'missing_values',
            'passed': True,
            'issues': [],
            'statistics': {}
        }

        # Calculate missing value statistics
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / len(df) * 100).round(2)

        results['statistics'] = {
            col: {
                'missing_count': int(count),
                'missing_pct': float(missing_pcts[col])
            }
            for col, count in missing_counts.items()
            if count > 0
        }

        # Check required columns
        if required_columns:
            for col in required_columns:
                if col in df.columns and df[col].isnull().any():
                    results['passed'] = False
                    results['issues'].append({
                        'type': 'required_column_has_nulls',
                        'column': col,
                        'null_count': int(df[col].isnull().sum())
                    })

        # Check maximum missing percentage
        for col in df.columns:
            missing_pct = missing_counts[col] / len(df)
            if missing_pct > max_missing_pct:
                results['passed'] = False
                results['issues'].append({
                    'type': 'excessive_missing_values',
                    'column': col,
                    'missing_pct': float(missing_pct * 100),
                    'threshold_pct': float(max_missing_pct * 100)
                })

        logger.info(f"Missing values validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_duplicates(self, df: pd.DataFrame,
                          unique_columns: Optional[List[str]] = None,
                          check_full_duplicates: bool = True) -> Dict:
        """
        Validate duplicate records.

        Args:
            df: DataFrame to validate
            unique_columns: Columns that should have unique values
            check_full_duplicates: Whether to check for fully duplicate rows

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating duplicates...")

        results = {
            'check': 'duplicates',
            'passed': True,
            'issues': []
        }

        # Check full duplicates
        if check_full_duplicates:
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                results['passed'] = False
                results['issues'].append({
                    'type': 'full_duplicate_rows',
                    'count': int(duplicate_rows)
                })

        # Check unique columns
        if unique_columns:
            for col in unique_columns:
                if col in df.columns:
                    duplicates = df[col].duplicated().sum()
                    if duplicates > 0:
                        results['passed'] = False
                        results['issues'].append({
                            'type': 'duplicate_values_in_unique_column',
                            'column': col,
                            'duplicate_count': int(duplicates)
                        })

        logger.info(f"Duplicates validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_value_ranges(self, df: pd.DataFrame,
                            range_rules: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Validate that numeric values fall within expected ranges.

        Args:
            df: DataFrame to validate
            range_rules: Dictionary mapping column names to (min, max) tuples

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating value ranges...")

        results = {
            'check': 'value_ranges',
            'passed': True,
            'issues': []
        }

        for col, (min_val, max_val) in range_rules.items():
            if col in df.columns:
                # Check minimum
                below_min = df[col] < min_val
                if below_min.any():
                    results['passed'] = False
                    results['issues'].append({
                        'type': 'values_below_minimum',
                        'column': col,
                        'min_allowed': min_val,
                        'min_found': float(df[col].min()),
                        'count': int(below_min.sum())
                    })

                # Check maximum
                above_max = df[col] > max_val
                if above_max.any():
                    results['passed'] = False
                    results['issues'].append({
                        'type': 'values_above_maximum',
                        'column': col,
                        'max_allowed': max_val,
                        'max_found': float(df[col].max()),
                        'count': int(above_max.sum())
                    })

        logger.info(f"Value ranges validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_categorical_values(self, df: pd.DataFrame,
                                   allowed_values: Dict[str, List[str]]) -> Dict:
        """
        Validate that categorical columns only contain allowed values.

        Args:
            df: DataFrame to validate
            allowed_values: Dictionary mapping column names to lists of allowed values

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating categorical values...")

        results = {
            'check': 'categorical_values',
            'passed': True,
            'issues': []
        }

        for col, allowed in allowed_values.items():
            if col in df.columns:
                # Find invalid values
                invalid_mask = ~df[col].isin(allowed) & df[col].notna()
                invalid_values = df.loc[invalid_mask, col].unique()

                if len(invalid_values) > 0:
                    results['passed'] = False
                    results['issues'].append({
                        'type': 'invalid_categorical_values',
                        'column': col,
                        'allowed_values': allowed,
                        'invalid_values': list(invalid_values),
                        'count': int(invalid_mask.sum())
                    })

        logger.info(f"Categorical values validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_date_formats(self, df: pd.DataFrame,
                            date_columns: List[str],
                            date_format: str = '%Y-%m-%d') -> Dict:
        """
        Validate date column formats and ranges.

        Args:
            df: DataFrame to validate
            date_columns: List of date column names
            date_format: Expected date format

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating date formats...")

        results = {
            'check': 'date_formats',
            'passed': True,
            'issues': []
        }

        for col in date_columns:
            if col in df.columns:
                # Try to parse dates
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        pd.to_datetime(df[col], format=date_format, errors='raise')

                    # Check for future dates (usually invalid)
                    date_col = pd.to_datetime(df[col])
                    future_dates = date_col > datetime.now()
                    if future_dates.any():
                        results['issues'].append({
                            'type': 'future_dates',
                            'column': col,
                            'count': int(future_dates.sum())
                        })

                except Exception as e:
                    results['passed'] = False
                    results['issues'].append({
                        'type': 'invalid_date_format',
                        'column': col,
                        'expected_format': date_format,
                        'error': str(e)
                    })

        logger.info(f"Date formats validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_email_format(self, df: pd.DataFrame,
                            email_column: str) -> Dict:
        """
        Validate email addresses format.

        Args:
            df: DataFrame to validate
            email_column: Name of email column

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating email formats...")

        results = {
            'check': 'email_format',
            'passed': True,
            'issues': []
        }

        if email_column in df.columns:
            # Simple email regex pattern
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

            invalid_emails = ~df[email_column].str.match(email_pattern, na=False)
            invalid_count = invalid_emails.sum()

            if invalid_count > 0:
                results['passed'] = False
                results['issues'].append({
                    'type': 'invalid_email_format',
                    'column': email_column,
                    'count': int(invalid_count),
                    'examples': list(df.loc[invalid_emails, email_column].head(5))
                })

        logger.info(f"Email format validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_referential_integrity(self, df1: pd.DataFrame,
                                      df2: pd.DataFrame,
                                      key_column: str,
                                      df1_name: str = 'df1',
                                      df2_name: str = 'df2') -> Dict:
        """
        Validate referential integrity between two DataFrames.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            key_column: Column to check for referential integrity
            df1_name: Name of first DataFrame (for reporting)
            df2_name: Name of second DataFrame (for reporting)

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating referential integrity between {df1_name} and {df2_name}...")

        results = {
            'check': 'referential_integrity',
            'passed': True,
            'issues': []
        }

        if key_column in df1.columns and key_column in df2.columns:
            # Find keys in df1 that don't exist in df2
            orphan_keys = ~df1[key_column].isin(df2[key_column])
            orphan_count = orphan_keys.sum()

            if orphan_count > 0:
                results['passed'] = False
                results['issues'].append({
                    'type': 'orphaned_keys',
                    'from_table': df1_name,
                    'to_table': df2_name,
                    'key_column': key_column,
                    'count': int(orphan_count),
                    'examples': list(df1.loc[orphan_keys, key_column].unique()[:5])
                })

        logger.info(f"Referential integrity validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_business_rules(self, df: pd.DataFrame,
                               rules: List[Dict[str, Any]]) -> Dict:
        """
        Validate custom business rules.

        Args:
            df: DataFrame to validate
            rules: List of business rule dictionaries with 'name', 'condition', and 'description'

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating business rules...")

        results = {
            'check': 'business_rules',
            'passed': True,
            'issues': []
        }

        for rule in rules:
            rule_name = rule['name']
            condition = rule['condition']
            description = rule.get('description', '')

            try:
                # Evaluate the condition
                violations = ~df.eval(condition)
                violation_count = violations.sum()

                if violation_count > 0:
                    results['passed'] = False
                    results['issues'].append({
                        'type': 'business_rule_violation',
                        'rule': rule_name,
                        'description': description,
                        'condition': condition,
                        'violation_count': int(violation_count)
                    })

            except Exception as e:
                results['passed'] = False
                results['issues'].append({
                    'type': 'business_rule_error',
                    'rule': rule_name,
                    'error': str(e)
                })

        logger.info(f"Business rules validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def validate_statistical_properties(self, df: pd.DataFrame,
                                       stats_rules: Dict[str, Dict[str, float]]) -> Dict:
        """
        Validate statistical properties of numeric columns.

        Args:
            df: DataFrame to validate
            stats_rules: Dictionary mapping column names to statistical rules
                        e.g., {'price': {'mean': 100, 'std': 50, 'tolerance': 0.2}}

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating statistical properties...")

        results = {
            'check': 'statistical_properties',
            'passed': True,
            'issues': []
        }

        for col, rules in stats_rules.items():
            if col in df.columns:
                actual_mean = df[col].mean()
                actual_std = df[col].std()

                # Check mean
                if 'mean' in rules and 'tolerance' in rules:
                    expected_mean = rules['mean']
                    tolerance = rules['tolerance']
                    if abs(actual_mean - expected_mean) / expected_mean > tolerance:
                        results['issues'].append({
                            'type': 'mean_deviation',
                            'column': col,
                            'expected_mean': expected_mean,
                            'actual_mean': float(actual_mean),
                            'tolerance': tolerance
                        })

                # Check std
                if 'std' in rules and 'tolerance' in rules:
                    expected_std = rules['std']
                    tolerance = rules['tolerance']
                    if abs(actual_std - expected_std) / expected_std > tolerance:
                        results['issues'].append({
                            'type': 'std_deviation',
                            'column': col,
                            'expected_std': expected_std,
                            'actual_std': float(actual_std),
                            'tolerance': tolerance
                        })

        if results['issues']:
            results['passed'] = False

        logger.info(f"Statistical properties validation: {'PASSED' if results['passed'] else 'FAILED'}")
        self.validation_results['checks'].append(results)
        if not results['passed']:
            self.validation_results['passed'] = False

        return results

    def get_validation_report(self) -> Dict:
        """
        Get comprehensive validation report.

        Returns:
            Dictionary with all validation results
        """
        return self.validation_results

    def save_report(self, filepath: str):
        """
        Save validation report to JSON file.

        Args:
            filepath: Path to save the report
        """
        with open(filepath, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        logger.info(f"Validation report saved to {filepath}")

    @staticmethod
    def _types_compatible(actual_type: str, expected_type: str) -> bool:
        """Check if actual and expected types are compatible."""
        type_mappings = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'string': ['object', 'str'],
            'datetime': ['datetime64', 'datetime64[ns]'],
            'bool': ['bool']
        }

        for expected, actuals in type_mappings.items():
            if expected_type.lower() in expected or expected_type in actuals:
                return any(a in actual_type for a in actuals)

        return actual_type == expected_type


def create_validation_summary(validation_results: Dict) -> str:
    """
    Create a human-readable summary of validation results.

    Args:
        validation_results: Validation results dictionary

    Returns:
        String summary
    """
    summary = []
    summary.append("=" * 60)
    summary.append("DATA VALIDATION SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Timestamp: {validation_results['timestamp']}")
    summary.append(f"Overall Status: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    summary.append("")

    for check in validation_results['checks']:
        summary.append(f"\nCheck: {check['check']}")
        summary.append(f"Status: {'PASSED' if check['passed'] else 'FAILED'}")

        if check.get('issues'):
            summary.append(f"Issues Found: {len(check['issues'])}")
            for issue in check['issues'][:3]:  # Show first 3 issues
                summary.append(f"  - {issue['type']}: {issue}")

        if check.get('statistics'):
            summary.append("Statistics:")
            for key, value in list(check['statistics'].items())[:3]:
                summary.append(f"  - {key}: {value}")

    summary.append("=" * 60)
    return "\n".join(summary)
