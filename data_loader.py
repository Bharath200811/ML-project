"""
UCI Online Retail Data Loader
=============================
Handles automatic format detection (.csv / .xlsx), data cleaning, 
cancellation filtering, unit price & quantity validation, and dataset summaries.

Author: Lead Machine Learning Engineer
"""

import os
import sys
import logging
import pandas as pd

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataLoader")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_retail_data(file_path: str) -> pd.DataFrame:
    """
    Loads and cleans UCI Online Retail transaction dataset.

    Cleaning Pipeline:
    ------------------
    1. Automatic format detection (.xlsx vs .csv ISO-8859-1)
    2. Drops rows with missing CustomerID
    3. Excludes cancelled invoices (InvoiceNo starting with 'C')
    4. Filters Quantity > 0 and UnitPrice > 0
    5. Computes Amount = Quantity * UnitPrice
    6. Converts InvoiceDate to datetime and normalizes columns to [CustomerID, Date, Amount]
    7. Logs dataset summary (row counts, unique customers, date range, total revenue)

    Parameters:
    -----------
    file_path : str
        Path to the real UCI Online Retail dataset (.csv or .xlsx).

    Returns:
    --------
    pd.DataFrame containing columns: CustomerID (str), Date (datetime64), Amount (float)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at path: {file_path}")

    logger.info(f"Loading raw retail dataset from '{file_path}'...")
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in [".xlsx", ".xls"]:
        df_raw = pd.read_excel(file_path)
    elif file_ext == ".csv":
        df_raw = pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        raise ValueError(f"Unsupported file format '{file_ext}'. Expected .csv or .xlsx/.xls.")

    initial_rows = len(df_raw)
    logger.info(f"Raw dataset loaded with {initial_rows:,} rows.")

    # 1. Drop missing CustomerID
    df_clean = df_raw.dropna(subset=["CustomerID"]).copy()

    # Format CustomerID cleanly as string (e.g. 17850.0 -> "17850")
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int).astype(str)

    # 2. Exclude cancelled invoices (InvoiceNo starting with 'C')
    df_clean["InvoiceNo"] = df_clean["InvoiceNo"].astype(str)
    df_clean = df_clean[~df_clean["InvoiceNo"].str.upper().str.startswith("C")]

    # 3. Filter Quantity > 0 and UnitPrice > 0
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

    # 4. Compute Amount = Quantity * UnitPrice
    df_clean["Amount"] = df_clean["Quantity"] * df_clean["UnitPrice"]

    # 5. Convert InvoiceDate to datetime
    date_col = "InvoiceDate" if "InvoiceDate" in df_clean.columns else "Date"
    df_clean["Date"] = pd.to_datetime(df_clean[date_col])

    # Select final standardized columns
    df_final = df_clean[["CustomerID", "Date", "Amount"]].sort_values(by="Date").reset_index(drop=True)
    final_rows = len(df_final)

    # 6. Data Summary Logging
    num_customers = df_final["CustomerID"].nunique()
    min_date = df_final["Date"].min()
    max_date = df_final["Date"].max()
    total_revenue = df_final["Amount"].sum()

    logger.info("=========================================================")
    logger.info("           UCI ONLINE RETAIL DATA SUMMARY               ")
    logger.info("=========================================================")
    logger.info(f"Initial Rows Loaded     : {initial_rows:,}")
    logger.info(f"Cleaned Valid Rows      : {final_rows:,} (Dropped {initial_rows - final_rows:,} invalid rows)")
    logger.info(f"Unique Customers Count  : {num_customers:,}")
    logger.info(f"Observation Date Range  : {min_date.date()} to {max_date.date()} ({(max_date - min_date).days} days)")
    logger.info(f"Total Combined Revenue  : ${total_revenue:,.2f} USD")
    logger.info("=========================================================\n")

    return df_final


def find_available_real_data_path() -> str:
    """
    Searches standard data paths for UCI Online Retail dataset files.
    Returns the path if found, or None if missing.
    """
    candidate_paths = [
        os.path.join(BASE_DIR, "data", "online_retail.csv"),
        os.path.join(BASE_DIR, "data", "Online Retail.xlsx"),
        os.path.join(BASE_DIR, "data", "OnlineRetail.csv"),
        os.path.join(BASE_DIR, "data", "Online Retail.csv"),
        os.path.join(BASE_DIR, "data", "online_retail_II.csv")
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None
