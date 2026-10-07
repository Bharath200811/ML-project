"""
Retail Customer Segmentation & Sales Forecasting Solution
==========================================================
Architected for enterprise retail analytics combining customer RFM behavioral
segmentation (K-Means) with temporal sales forecasting (Random Forest Regressor).

Supports real UCI Online Retail dataset with automated format detection, skew log-transformations,
and full daily calendar reindexing.

Author: Lead Machine Learning Engineer
Date: 2026-10-07
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from data_loader import load_retail_data, find_available_real_data_path
from generate_dataset import generate_synthetic_transactions
from forecasting import prepare_daily_data, split_train_test, train_model, test_model

# Toggle to use real UCI Online Retail data if present (with fallback to synthetic data)
USE_REAL_DATA = True

# Setup clean logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RetailIntelligence")

# Set random seeds for strict reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Matplotlib visual style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Absolute directory paths relative to main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "synthetic_retail_transactions.csv")


# =====================================================================
# STEP 1: Data Ingestion & Routing
# =====================================================================
def get_retail_dataset(use_real_data: bool = USE_REAL_DATA) -> pd.DataFrame:
    """
    Informs dataset selection between real UCI dataset and synthetic fallback.
    Prints explicit dataset path and directory audit when falling back.
    """
    if use_real_data:
        real_data_path = find_available_real_data_path()
        if real_data_path:
            logger.info(f"Using REAL dataset: {real_data_path}")
            return load_retail_data(real_data_path)
        else:
            logger.warning("=========================================================")
            logger.warning("REAL DATASET NOT FOUND!")
            logger.warning(
                "USE_REAL_DATA=True, but expected real dataset file "
                "(data/online_retail.csv or data/Online Retail.xlsx) was not found."
            )
            
            # Audit files inside data/ folder
            if os.path.exists(DATA_DIR):
                data_files = os.listdir(DATA_DIR)
                logger.warning(f"Files currently inside 'data/' folder: {data_files if data_files else '[Empty Directory]'}")
            else:
                logger.warning("Directory 'data/' does not exist.")
                
            logger.warning("Falling back to SYNTHETIC dataset...")
            logger.warning("=========================================================\n")

    # Synthetic fallback
    logger.info("Using SYNTHETIC dataset")
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        logger.info(f"Synthetic dataset missing at '{SYNTHETIC_DATA_PATH}'. Generating dataset now...")
        return generate_synthetic_transactions(output_path=SYNTHETIC_DATA_PATH)

    return pd.read_csv(SYNTHETIC_DATA_PATH, parse_dates=["Date"])


# =====================================================================
# STEP 2: Track 1 - Customer RFM & K-Means Segmentation
# =====================================================================
def run_rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes RFM feature extraction (Recency calculated against max_date + 1 day),
    applies np.log1p transformation to handle heavily skewed Frequency and Monetary distributions,
    fits KMeans(n_clusters=3, random_state=42, n_init=10), and maps behavioral persona profiles.
    
    Returns:
    --------
    pd.DataFrame with customer-level RFM metrics, scaled features, Cluster IDs, and Segment Labels.
    """
    logger.info("Executing Track 1: RFM Customer Segmentation...")
    
    # Recency computed against max_date + 1 day
    max_date = df["Date"].max()
    reference_date = max_date + pd.Timedelta(days=1)
    
    rfm_df = df.groupby("CustomerID").agg(
        Recency=("Date", lambda x: (reference_date - x.max()).days),
        Frequency=("Date", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()
    
    # Apply np.log1p transformation to mitigate severe right skew before scaling
    rfm_df["Recency_log"] = np.log1p(rfm_df["Recency"])
    rfm_df["Frequency_log"] = np.log1p(rfm_df["Frequency"])
    rfm_df["Monetary_log"] = np.log1p(rfm_df["Monetary"])
    
    # Feature Scaling using StandardScaler on log-transformed features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_df[["Recency_log", "Frequency_log", "Monetary_log"]])
    
    # Fit K-Means Clustering (n_clusters=3, random_state=42, n_init=10)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    rfm_df["Cluster"] = kmeans.fit_predict(scaled_features)
    
    # Compute Cluster-Level Medians to dynamically map behavioral persona labels
    cluster_medians = rfm_df.groupby("Cluster")["Monetary"].median().sort_values()
    
    # Map cluster IDs by median monetary spend order
    cluster_order = cluster_medians.index.tolist()
    label_map = {
        cluster_order[0]: "At-Risk / Low Spender",
        cluster_order[1]: "Regular Customer",
        cluster_order[2]: "VIP / High Spender"
    }
    
    rfm_df["Segment"] = rfm_df["Cluster"].map(label_map)
    
    # Display 3 Segment Summary Profile Table
    summary = rfm_df.groupby("Segment").agg(
        Customer_Count=("CustomerID", "count"),
        Recency_Median=("Recency", "median"),
        Frequency_Median=("Frequency", "median"),
        Monetary_Median=("Monetary", "median"),
        Total_Segment_Spend=("Monetary", "sum")
    ).reindex(["VIP / High Spender", "Regular Customer", "At-Risk / Low Spender"])
    
    logger.info("RFM Segmentation completed successfully.")
    logger.info("\n=== CUSTOMER SEGMENT PROFILE SUMMARY ===\n" + summary.to_string())
    
    return rfm_df


# =====================================================================
# STEP 3: High-Impact Visualization Dashboard
# =====================================================================
def plot_retail_intelligence_dashboard(
    rfm_df: pd.DataFrame,
    forecast_results: dict,
    output_path: str = os.path.join(OUTPUT_DIR, "retail_intelligence_dashboard.png")
):
    """
    Generates a 4-panel visual dashboard for customer segmentation & daily sales forecasting.
    """
    logger.info(f"Generating visual intelligence dashboard saved to '{output_path}'...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Retail Customer Segmentation & Sales Forecasting Intelligence Dashboard", fontsize=18, fontweight='bold', y=0.98)
    
    palette = {"VIP / High Spender": "#2ecc71", "Regular Customer": "#3498db", "At-Risk / Low Spender": "#e74c3c"}
    
    # 1. Segmentation Scatter: Recency vs Monetary Spend
    ax1 = axes[0, 0]
    sns.scatterplot(
        data=rfm_df,
        x="Recency",
        y="Monetary",
        hue="Segment",
        palette=palette,
        style="Segment",
        s=80,
        alpha=0.9,
        ax=ax1
    )
    ax1.set_title("Customer RFM Segmentation (Recency vs Total Spend)", fontsize=13, fontweight='semibold')
    ax1.set_xlabel("Recency (Days Since Last Purchase)")
    ax1.set_ylabel("Monetary Value ($ USD Total Spend)")
    ax1.legend(title="Customer Segment", frameon=True)
    
    # 2. Segment Monetary Spend Distributions (Boxplot with hue=Segment and legend=False)
    ax2 = axes[0, 1]
    sns.boxplot(
        data=rfm_df,
        x="Segment",
        y="Monetary",
        palette=palette,
        hue="Segment",
        legend=False,
        ax=ax2,
        boxprops=dict(alpha=0.8)
    )
    ax2.set_title("Monetary Spend Distribution Across Segments", fontsize=13, fontweight='semibold')
    ax2.set_xlabel("Customer Segment")
    ax2.set_ylabel("Monetary Value ($ USD)")
    
    # 3. Forecasting: Actual vs Predicted Daily Sales on Holdout Test Set
    ax3 = axes[1, 0]
    test_dates = forecast_results["test_df"]["Date"]
    ax3.plot(test_dates, forecast_results["y_test"], label="Actual Daily Sales", color="#2c3e50", linewidth=2, marker='o', markersize=3)
    ax3.plot(test_dates, forecast_results["y_pred"], label="Forecasted Sales (Random Forest)", color="#e67e22", linewidth=2, linestyle="--", marker='s', markersize=3)
    ax3.set_title(f"Daily Sales Forecasting (Holdout Test Set)\nMAE: ${forecast_results['mae']:.2f} USD/day | R²: {forecast_results['r2']:.3f}", fontsize=13, fontweight='semibold')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Daily Aggregated Sales ($ USD)")
    ax3.legend(frameon=True)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # 4. Feature Importance Bar Plot (with hue=y_variable and legend=False)
    ax4 = axes[1, 1]
    importances = forecast_results["feature_importances"]
    sns.barplot(
        x=importances.values,
        y=importances.index,
        palette="viridis",
        hue=importances.index,
        legend=False,
        ax=ax4
    )
    ax4.set_title("Random Forest Predictor Importance", fontsize=13, fontweight='semibold')
    ax4.set_xlabel("Relative Feature Importance Score")
    ax4.set_ylabel("Predictor Variable")
    for i, v in enumerate(importances.values):
        ax4.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold')
    ax4.set_xlim(0, max(importances.values) * 1.15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Dashboard saved successfully to '{output_path}'.")


# =====================================================================
# PIPELINE EXECUTION ENTRY POINT
# =====================================================================
def main():
    logger.info("=========================================================")
    logger.info("Starting Retail Customer Segmentation & Sales Forecasting")
    logger.info("=========================================================")
    
    # Ensure outputs directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Data Ingestion from Real Dataset (or synthetic fallback)
    transactions_df = get_retail_dataset(use_real_data=USE_REAL_DATA)
    
    # 2. Track 1: Customer Segmentation
    rfm_df = run_rfm_segmentation(transactions_df)
    rfm_output_path = os.path.join(OUTPUT_DIR, "customer_rfm_segments.csv")
    rfm_df.to_csv(rfm_output_path, index=False)
    logger.info(f"Saved RFM customer segmentation results to '{rfm_output_path}'.")
    
    # 3. Track 2: Daily Sales Forecasting (Separated Pipeline Steps)
    daily_clean = prepare_daily_data(transactions_df)
    X_train, y_train, X_test, y_test, train_df, test_df = split_train_test(daily_clean)
    
    # Training Part Only
    rf_model = train_model(X_train, y_train)
    
    # Testing Part Only
    y_pred, mae, r2 = test_model(rf_model, X_test, y_test)
    
    feature_cols = X_test.columns.tolist()
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    forecast_results = {
        "model": rf_model,
        "mae": mae,
        "r2": r2,
        "daily_clean": daily_clean,
        "train_df": train_df,
        "test_df": test_df,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importances": importances
    }
    
    # 4. Generate Visualizations
    dashboard_output_path = os.path.join(OUTPUT_DIR, "retail_intelligence_dashboard.png")
    plot_retail_intelligence_dashboard(rfm_df, forecast_results, output_path=dashboard_output_path)
    
    logger.info("\nPipeline execution completed successfully with zero errors!")


if __name__ == "__main__":
    main()
