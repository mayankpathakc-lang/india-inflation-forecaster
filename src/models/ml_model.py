import xgboost as xgb
import numpy as np
import pandas as pd
import sys
import os

# Ensure we can import features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from features import generate_features
except ImportError:
    pass

def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost model on lag/rolling features.
    """
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def get_residuals_std(model, X_train, y_train):
    """
    Calculate residual-based standard error for prediction intervals.
    """
    preds = model.predict(X_train)
    residuals = y_train - preds
    return np.std(residuals)

def generate_features_for_df(df):
    """Replicates features.py feature engineering on a DataFrame without loading from CSV."""
    df = df.copy()
    target = 'YoY_Inflation'
    
    # Generate Lag features
    lags = [1, 2, 3, 12]
    for lag in lags:
        df[f'YoY_Lag_{lag}'] = df[target].shift(lag)
        
    # Generate Rolling statistics (shifted by 1 to prevent data leakage)
    windows = [3, 6, 12]
    for window in windows:
        df[f'YoY_Rolling_Mean_{window}'] = df[target].shift(1).rolling(window=window).mean()
        df[f'YoY_Rolling_Std_{window}'] = df[target].shift(1).rolling(window=window).std()
        
    # Generate Lag features for Exogenous variables
    exog_cols = ['Repo_Rate', 'Crude_Oil_YoY_Change', 'USD_INR_YoY_Change']
    for col in exog_cols:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
                
    # Cyclical encoding for Month of year
    months = df.index.month
    df['Month_Sin'] = np.sin(2 * np.pi * months / 12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * months / 12.0)
    
    return df

def recursive_forecast_xgboost(model, history_df, n_steps=12, feature_cols=None):
    """
    Implement recursive multi-step forecasting with support for exogenous variables.
    Carries forward the last known value of exogenous variables.
    """
    current_df = history_df.copy()
    forecasts = []
    
    target_col = 'YoY_Inflation'
    exog_cols = ['Repo_Rate', 'Crude_Oil_YoY_Change', 'USD_INR_YoY_Change']
    cols_to_keep = [target_col] + [c for c in exog_cols if c in current_df.columns]
    
    current_df = current_df[cols_to_keep]
    
    for _ in range(n_steps):
        next_date = current_df.index[-1] + pd.DateOffset(months=1)
        
        # Carry forward the last known value of target and exogenous variables
        next_row = {target_col: 0.0}
        for col in exog_cols:
            if col in current_df.columns:
                next_row[col] = current_df[col].iloc[-1]
                
        next_df = pd.DataFrame(next_row, index=[next_date])
        current_df = pd.concat([current_df, next_df])
        
        # Calculate features on the expanded DataFrame
        features_df = generate_features_for_df(current_df)
        X_next = features_df[feature_cols].iloc[-1:]
        
        # Predict next step
        pred = model.predict(X_next)[0]
        forecasts.append(pred)
        
        # Update target in dataframe
        current_df.loc[next_date, target_col] = pred
        
    return forecasts
