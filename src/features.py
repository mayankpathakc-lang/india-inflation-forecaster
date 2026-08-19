import pandas as pd
import numpy as np

def generate_features(input_path="data/processed/india_inflation.csv"):
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    target = 'YoY_Inflation'
    
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in the dataset.")
        
    # Generate target Lag features
    lags = [1, 2, 3, 12]
    for lag in lags:
        df[f'YoY_Lag_{lag}'] = df[target].shift(lag)
        
    # Generate Rolling statistics of the target (shifted by 1 to prevent data leakage)
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
    
    # Drop NaNs introduced by lags and rolling windows
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    features_df = generate_features()
    print("Features generated successfully.")
    print("Shape:", features_df.shape)
    print("Columns:", features_df.columns.tolist())
