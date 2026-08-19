import pandas as pd
import numpy as np
import os
import joblib
import sys

# Ensure we can import modules from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import fetch_all_data
from preprocess import preprocess_data
from features import generate_features
from models.baseline_arima import train_sarima, forecast_sarima
from models.ml_model import train_xgboost, get_residuals_std, recursive_forecast_xgboost

def main():
    print("=== Step 1: Fetching all Macroeconomic Data ===")
    fetch_all_data()
    
    print("\n=== Step 2: Preprocessing ===")
    preprocess_data()
    
    print("\n=== Step 3: Feature Engineering ===")
    df = generate_features()
    
    target_col = 'YoY_Inflation'
    exog_names = ['Repo_Rate', 'Crude_Oil_YoY_Change', 'USD_INR_YoY_Change']
    exog_cols_present = [c for c in exog_names if c in df.columns]
    print(f"Exogenous variables detected: {exog_cols_present}")
    
    # Feature columns for the ML model (only lags, rolling target, and lags of exog)
    feature_cols = [c for c in df.columns if (
        c.startswith('YoY_Lag') or 
        c.startswith('YoY_Rolling') or 
        c.startswith('Month_') or 
        any(c.startswith(exog + '_Lag_') for exog in exog_cols_present)
    )]
    print(f"ML Model features ({len(feature_cols)}): {feature_cols}")
    
    # 4. Split data chronologically
    from models.evaluate import split_data, calculate_metrics
    train_df, test_df = split_data(df, train_ratio=0.8)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 5. Train SARIMAX (Baseline)
    print("\nTraining SARIMAX baseline...")
    X_train_sarima = train_df[exog_cols_present] if exog_cols_present else None
    X_test_sarima = test_df[exog_cols_present] if exog_cols_present else None
    
    sarima_model = train_sarima(train_df[target_col], X=X_train_sarima)
    
    # Evaluate SARIMAX
    sarima_preds, _ = forecast_sarima(sarima_model, n_steps=len(test_df), X=X_test_sarima)
    sarima_metrics = calculate_metrics(test_df[target_col], sarima_preds)
    print("SARIMAX Metrics:", sarima_metrics)
    
    # 6. Train XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = train_xgboost(train_df[feature_cols], train_df[target_col])
    
    # Evaluate XGBoost recursively
    xgb_preds = recursive_forecast_xgboost(
        xgb_model, 
        history_df=train_df, 
        n_steps=len(test_df), 
        feature_cols=feature_cols
    )
    xgb_metrics = calculate_metrics(test_df[target_col], xgb_preds)
    print("XGBoost Metrics:", xgb_metrics)
    
    # 7. Future Forecasting (Next 12 months)
    print("\nForecasting next 12 months...")
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    
    # Carry forward exogenous values for the next 12 months
    if exog_cols_present:
        future_exog = pd.DataFrame(index=future_dates)
        for col in exog_cols_present:
            future_exog[col] = df[col].iloc[-1]
    else:
        future_exog = None
        
    # Retrain on full data
    X_full_sarima = df[exog_cols_present] if exog_cols_present else None
    sarima_full = train_sarima(df[target_col], X=X_full_sarima)
    future_sarima, sarima_conf = forecast_sarima(sarima_full, n_steps=12, X=future_exog)
    
    xgb_full = train_xgboost(df[feature_cols], df[target_col])
    future_xgb = recursive_forecast_xgboost(
        xgb_full, 
        history_df=df, 
        n_steps=12, 
        feature_cols=feature_cols
    )
    xgb_std = get_residuals_std(xgb_full, df[feature_cols], df[target_col])
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'SARIMA_Forecast': future_sarima,
        'SARIMA_Lower': sarima_conf[:, 0],
        'SARIMA_Upper': sarima_conf[:, 1],
        'XGB_Forecast': future_xgb,
        'XGB_Lower': np.array(future_xgb) - 1.96 * xgb_std,
        'XGB_Upper': np.array(future_xgb) + 1.96 * xgb_std
    })
    
    os.makedirs('data/forecasts', exist_ok=True)
    forecast_df.to_csv('data/forecasts/future_forecasts.csv', index=False)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    import pickle
    with open('models/sarima_model.pkl', 'wb') as f:
        pickle.dump(sarima_full, f)
    joblib.dump(xgb_full, 'models/xgb_model.joblib')
    
    # Save metrics
    metrics_df = pd.DataFrame([
        {'Model': 'SARIMA', **sarima_metrics},
        {'Model': 'XGBoost', **xgb_metrics}
    ])
    metrics_df.to_csv('data/forecasts/metrics.csv', index=False)
    
    print("Done! Artifacts saved.")

if __name__ == "__main__":
    main()
