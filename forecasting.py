"""
Daily Sales Forecasting Module
==============================
Provides feature preparation, train/test splitting, model training,
model serialization, and holdout evaluation.

Author: Lead Machine Learning Engineer
Date: 2026-10-07
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SalesForecasting")

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "rf_model.pkl")


# =====================================================================
# ===== DATA PREPARATION PART =====
# =====================================================================
def prepare_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily sales, reindexes across a full daily calendar filling 0s,
    engineers calendar and 1-day / 7-day lag features, and drops NaN lookback rows.

    Parameters:
    -----------
    df : pd.DataFrame
        Transaction records containing Date and Amount.

    Returns:
    --------
    pd.DataFrame containing cleaned daily dataset (Date, DailySales, DayOfWeek, Month, Lag_1_Day, Lag_7_Day).
    """
    logger.info("Preparing daily sales calendar and lag features...")
    
    # 1. Normalize Date to calendar day and aggregate DailySales sum
    df_norm = df.copy()
    df_norm["DateOnly"] = df_norm["Date"].dt.floor("D")
    daily_agg = df_norm.groupby("DateOnly")["Amount"].sum().reset_index()
    
    # 2. Reindex across complete daily calendar from min_date to max_date
    min_date = daily_agg["DateOnly"].min()
    max_date = daily_agg["DateOnly"].max()
    full_calendar = pd.date_range(start=min_date, end=max_date, freq="D")
    
    daily_df = daily_agg.set_index("DateOnly").reindex(full_calendar, fill_value=0.0).reset_index()
    daily_df.columns = ["Date", "DailySales"]
    daily_df = daily_df.sort_values(by="Date").reset_index(drop=True)
    
    # 3. Feature Engineering: DayOfWeek, Month, Lag_1_Day, Lag_7_Day
    daily_df["DayOfWeek"] = daily_df["Date"].dt.dayofweek
    daily_df["Month"] = daily_df["Date"].dt.month
    daily_df["Lag_1_Day"] = daily_df["DailySales"].shift(1)
    daily_df["Lag_7_Day"] = daily_df["DailySales"].shift(7)
    
    # Drop undefined lookback rows
    daily_clean = daily_df.dropna().reset_index(drop=True)
    logger.info(
        f"Cleaned daily calendar dataset prepared: {len(daily_clean)} observations "
        f"spanning {daily_clean['Date'].min().date()} to {daily_clean['Date'].max().date()}."
    )
    
    return daily_clean


# =====================================================================
# ===== TRAIN / TEST SPLIT PART =====
# =====================================================================
def split_train_test(daily_clean: pd.DataFrame, train_ratio: float = 0.8):
    """
    Enforces a strict sequential chronological split (no shuffling) into train and test sets.

    Parameters:
    -----------
    daily_clean : pd.DataFrame
        Prepared daily dataset with features and target.
    train_ratio : float
        Proportion of observations reserved for training (default 0.8).

    Returns:
    --------
    X_train, y_train, X_test, y_test, train_df, test_df
    """
    logger.info("Executing sequential chronological train/test split...")
    
    feature_cols = ["DayOfWeek", "Month", "Lag_1_Day", "Lag_7_Day"]
    target_col = "DailySales"
    
    split_idx = int(len(daily_clean) * train_ratio)
    train_df = daily_clean.iloc[:split_idx]
    test_df = daily_clean.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    train_start, train_end = train_df['Date'].min().date(), train_df['Date'].max().date()
    test_start, test_end = test_df['Date'].min().date(), test_df['Date'].max().date()
    
    logger.info("=========================================================")
    logger.info("            CHRONOLOGICAL TRAIN/TEST SPLIT SUMMARY      ")
    logger.info("=========================================================")
    logger.info(f"Training Set Size   : {len(train_df)} days ({train_start} to {train_end})")
    logger.info(f"Testing Set Size    : {len(test_df)} days ({test_start} to {test_end})")
    logger.info("=========================================================\n")
    
    return X_train, y_train, X_test, y_test, train_df, test_df


# =====================================================================
# ===== TRAINING PART =====
# =====================================================================
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    TRAINING PART ONLY:
    Fits RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    and serializes the trained model to outputs/rf_model.pkl.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Predictor features for training.
    y_train : pd.Series
        Target daily sales values for training.

    Returns:
    --------
    RandomForestRegressor
        Trained model instance.
    """
    logger.info("===== TRAINING PART: Fitting RandomForestRegressor =====")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # Ensure outputs directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    logger.info(f"Trained Random Forest model successfully saved to '{MODEL_SAVE_PATH}'.")
    
    return model


# =====================================================================
# ===== TESTING PART =====
# =====================================================================
def test_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    """
    TESTING PART ONLY:
    Predicts on holdout test set, calculates evaluation metrics (MAE and R^2),
    and logs performance breakdown.

    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model instance.
    X_test : pd.DataFrame
        Predictor features for test set.
    y_test : pd.Series
        Ground truth daily sales target for test set.

    Returns:
    --------
    tuple: (y_pred, mae, r2)
    """
    logger.info("===== TESTING PART: Evaluating Model on Holdout Set =====")
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info("\n=== FORECAST MODEL EVALUATION METRICS ===")
    logger.info(f"Holdout Mean Absolute Error (MAE): ${mae:.2f} USD/day")
    logger.info(f"Holdout Coefficient of Determination (R^2): {r2:.4f}")
    
    feature_cols = X_test.columns.tolist()
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info("\nFeature Importance Breakdown:\n" + importances.to_string())
    
    return y_pred, mae, r2
