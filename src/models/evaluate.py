import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def split_data(df, train_ratio=0.8):
    """Splits data chronologically."""
    train_size = int(len(df) * train_ratio)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    return train, test

def calculate_metrics(actual, predicted):
    """Computes MAE, RMSE, MAPE."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    mask = actual != 0
    if np.any(mask):
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan
        
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def walk_forward_validation(df, target_col, train_ratio=0.8):
    # This might be tricky to unify for SARIMA and XGBoost since they need different inputs.
    # It might be easier if we evaluate them directly in their respective files or forecast.py.
    pass
