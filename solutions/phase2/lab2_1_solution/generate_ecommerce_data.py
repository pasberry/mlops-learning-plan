"""
E-commerce Data Generator
Generates synthetic e-commerce data for ETL pipeline testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os


class EcommerceDataGenerator:
    """Generate synthetic e-commerce data for testing."""

    def __init__(self, seed=42):
        """Initialize the data generator with a random seed."""
        random.seed(seed)
        np.random.seed(seed)

        # Product catalog
        self.categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Toys', 'Sports']
        self.products = {
            'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Camera', 'Smartwatch'],
            'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes', 'Hat'],
            'Home & Garden': ['Lamp', 'Chair', 'Table', 'Plant', 'Rug', 'Mirror'],
            'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Cookbook', 'Biography', 'Comic'],
            'Toys': ['Action Figure', 'Board Game', 'Puzzle', 'Doll', 'Building Blocks', 'RC Car'],
            'Sports': ['Basketball', 'Tennis Racket', 'Yoga Mat', 'Dumbbells', 'Running Shoes', 'Bicycle']
        }

        # Price ranges by category
        self.price_ranges = {
            'Electronics': (50, 2000),
            'Clothing': (10, 200),
            'Home & Garden': (15, 500),
            'Books': (5, 50),
            'Toys': (5, 100),
            'Sports': (10, 1000)
        }

        # Customer locations
        self.countries = ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Japan']
        self.cities = {
            'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'UK': ['London', 'Manchester', 'Birmingham', 'Liverpool', 'Leeds'],
            'Canada': ['Toronto', 'Montreal', 'Vancouver', 'Calgary', 'Ottawa'],
            'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne'],
            'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'],
            'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'],
            'Japan': ['Tokyo', 'Osaka', 'Yokohama', 'Nagoya', 'Sapporo']
        }

        # Payment methods
        self.payment_methods = ['Credit Card', 'PayPal', 'Debit Card', 'Bank Transfer', 'Apple Pay']

        # Order statuses
        self.statuses = ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
        self.status_weights = [0.1, 0.15, 0.25, 0.45, 0.05]

    def generate_customer_id(self):
        """Generate a random customer ID."""
        return f"CUST-{random.randint(1000, 9999)}"

    def generate_order_id(self):
        """Generate a random order ID."""
        return f"ORD-{random.randint(100000, 999999)}"

    def generate_product_info(self):
        """Generate random product information."""
        category = random.choice(self.categories)
        product_name = random.choice(self.products[category])
        price_min, price_max = self.price_ranges[category]
        price = round(random.uniform(price_min, price_max), 2)

        return {
            'category': category,
            'product_name': product_name,
            'price': price
        }

    def generate_location(self):
        """Generate random customer location."""
        country = random.choice(self.countries)
        city = random.choice(self.cities[country])
        return country, city

    def generate_orders(self, num_orders=1000, start_date=None, end_date=None):
        """
        Generate synthetic order data.

        Args:
            num_orders: Number of orders to generate
            start_date: Start date for orders (default: 30 days ago)
            end_date: End date for orders (default: today)

        Returns:
            DataFrame with order data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        orders = []

        for _ in range(num_orders):
            # Generate order timestamp
            time_delta = end_date - start_date
            random_seconds = random.randint(0, int(time_delta.total_seconds()))
            order_date = start_date + timedelta(seconds=random_seconds)

            # Generate product info
            product_info = self.generate_product_info()

            # Generate location
            country, city = self.generate_location()

            # Generate quantity (1-5 items)
            quantity = random.randint(1, 5)

            # Calculate total
            total_amount = round(product_info['price'] * quantity, 2)

            # Add some noise to create data quality issues
            # 5% chance of missing data
            if random.random() < 0.05:
                customer_id = None if random.random() < 0.3 else self.generate_customer_id()
                total_amount = None if random.random() < 0.3 else total_amount
            else:
                customer_id = self.generate_customer_id()

            # 2% chance of negative values (data quality issue)
            if random.random() < 0.02:
                quantity = -abs(quantity)
            if random.random() < 0.02:
                total_amount = -abs(total_amount) if total_amount else total_amount

            order = {
                'order_id': self.generate_order_id(),
                'customer_id': customer_id,
                'order_date': order_date.strftime('%Y-%m-%d %H:%M:%S'),
                'product_category': product_info['category'],
                'product_name': product_info['product_name'],
                'quantity': quantity,
                'price': product_info['price'],
                'total_amount': total_amount,
                'country': country,
                'city': city,
                'payment_method': random.choice(self.payment_methods),
                'status': np.random.choice(self.statuses, p=self.status_weights)
            }

            orders.append(order)

        return pd.DataFrame(orders)

    def generate_customer_data(self, num_customers=500):
        """
        Generate synthetic customer data.

        Args:
            num_customers: Number of customers to generate

        Returns:
            DataFrame with customer data
        """
        customers = []

        for i in range(num_customers):
            # Generate registration date (last 2 years)
            registration_date = datetime.now() - timedelta(days=random.randint(0, 730))

            # Generate location
            country, city = self.generate_location()

            # Generate email (some with invalid formats for data quality testing)
            if random.random() < 0.95:
                email = f"customer{i}@example.com"
            else:
                # Invalid email format
                email = f"customer{i}@invalid" if random.random() < 0.5 else f"customer{i}"

            # Generate age (some outliers)
            if random.random() < 0.95:
                age = random.randint(18, 75)
            else:
                # Outlier ages
                age = random.randint(1, 120)

            customer = {
                'customer_id': f"CUST-{1000 + i}",
                'email': email,
                'registration_date': registration_date.strftime('%Y-%m-%d'),
                'country': country,
                'city': city,
                'age': age,
                'is_premium': random.choice([True, False])
            }

            customers.append(customer)

        return pd.DataFrame(customers)

    def save_to_csv(self, df, filepath, include_header=True):
        """Save DataFrame to CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, header=include_header)
        print(f"Saved {len(df)} records to {filepath}")

    def save_to_json(self, df, filepath, orient='records'):
        """Save DataFrame to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_json(filepath, orient=orient, date_format='iso', indent=2)
        print(f"Saved {len(df)} records to {filepath}")


def main():
    """Main function to generate sample data."""
    print("Generating e-commerce data...")

    # Initialize generator
    generator = EcommerceDataGenerator(seed=42)

    # Create output directory
    output_dir = '/tmp/ecommerce_data'
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save orders
    print("\n1. Generating orders...")
    orders = generator.generate_orders(num_orders=1000)
    generator.save_to_csv(orders, f'{output_dir}/raw_orders.csv')
    generator.save_to_json(orders, f'{output_dir}/raw_orders.json')

    # Generate and save customers
    print("\n2. Generating customers...")
    customers = generator.generate_customer_data(num_customers=500)
    generator.save_to_csv(customers, f'{output_dir}/raw_customers.csv')
    generator.save_to_json(customers, f'{output_dir}/raw_customers.json')

    # Print summary statistics
    print("\n" + "="*60)
    print("DATA GENERATION SUMMARY")
    print("="*60)
    print(f"\nOrders Generated: {len(orders)}")
    print(f"Customers Generated: {len(customers)}")
    print(f"\nOutput Directory: {output_dir}")
    print("\nFiles created:")
    print("  - raw_orders.csv")
    print("  - raw_orders.json")
    print("  - raw_customers.csv")
    print("  - raw_customers.json")

    # Show sample data
    print("\n" + "="*60)
    print("SAMPLE ORDERS DATA")
    print("="*60)
    print(orders.head(5))

    print("\n" + "="*60)
    print("DATA QUALITY ISSUES (INTENTIONAL)")
    print("="*60)
    print(f"Orders with missing values: {orders.isnull().any(axis=1).sum()}")
    print(f"Orders with negative quantities: {(orders['quantity'] < 0).sum()}")
    print(f"Orders with negative amounts: {(orders['total_amount'] < 0).sum() if orders['total_amount'].notna().any() else 0}")
    print(f"Customers with invalid emails: {(~customers['email'].str.contains('@.*\\.', na=False)).sum()}")
    print(f"Customers with outlier ages: {((customers['age'] < 18) | (customers['age'] > 100)).sum()}")


if __name__ == "__main__":
    main()
