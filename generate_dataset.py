"""
Synthetic Retail Transaction Generator
======================================
Generates synthetic retail transaction records across customers and dates
with weekend spend dynamics, promo boosts, and exponential noise.

Author: Lead Machine Learning Engineer
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataGenerator")

# Fixed random seed for strict reproducibility
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_retail_transactions.csv")


def generate_synthetic_transactions(
    num_rows: int = 1200,
    num_customers: int = 150,
    num_days: int = 180,
    start_date: str = "2025-01-01",
    output_path: str = DEFAULT_DATA_PATH
) -> pd.DataFrame:
    """
    Generates synthetic retail transaction records with realistic spend dynamics
    and saves to CSV.

    Parameters:
    -----------
    num_rows : int
        Total number of transaction records (1,200).
    num_customers : int
        Number of unique customer IDs (150).
    num_days : int
        Observation window in days (180 days).
    start_date : str
        Simulation start date ("2025-01-01").
    output_path : str
        Path where CSV will be saved.

    Returns:
    --------
    pd.DataFrame containing CustomerID, Date, Amount, IsPromo
    """
    logger.info("Initializing synthetic transaction data generation...")
    np.random.seed(RANDOM_STATE)

    customer_ids = [f"CUST_{i:03d}" for i in range(1, num_customers + 1)]
    date_range = pd.date_range(start=start_date, periods=num_days, freq="D")

    # Sample random dates and customers to construct 1,200 transactions
    sampled_dates = np.random.choice(date_range, size=num_rows, replace=True)
    sampled_customers = np.random.choice(customer_ids, size=num_rows, replace=True)

    # 30% promotion event probability
    is_promo = np.random.binomial(n=1, p=0.30, size=num_rows)

    # Spend modeling: $60 base spend on weekends vs $30 on weekdays + $15 promo boost + exponential noise (scale 10)
    amounts = []
    for dt, promo in zip(sampled_dates, is_promo):
        # Convert np.datetime64 to pd.Timestamp to safely access .weekday()
        day_of_week = pd.Timestamp(dt).weekday()
        is_weekend = day_of_week in [5, 6]

        base_spend = 60.0 if is_weekend else 30.0
        promo_boost = 15.0 if promo == 1 else 0.0
        exponential_noise = np.random.exponential(scale=10.0)

        total_amount = round(base_spend + promo_boost + exponential_noise, 2)
        amounts.append(total_amount)

    df = pd.DataFrame({
        "CustomerID": sampled_customers,
        "Date": pd.to_datetime(sampled_dates),
        "Amount": amounts,
        "IsPromo": is_promo
    })

    # Sort chronologically
    df = df.sort_values(by="Date").reset_index(drop=True)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save to CSV (format Date as YYYY-MM-DD string)
    df_to_save = df.copy()
    df_to_save["Date"] = df_to_save["Date"].dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(output_path, index=False)

    logger.info(
        f"Generated {len(df)} transactions across {df['CustomerID'].nunique()} customers "
        f"spanning {df['Date'].min().date()} to {df['Date'].max().date()}."
    )
    logger.info(f"Successfully saved synthetic dataset to '{output_path}'.")

    return df


if __name__ == "__main__":
    generate_synthetic_transactions()
