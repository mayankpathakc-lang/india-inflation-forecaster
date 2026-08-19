import pandas as pd
from statsmodels.tsa.stattools import adfuller

def preprocess_data(input_path="data/raw/india_cpi.csv", output_path="data/processed/india_inflation.csv"):
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Resample to Monthly Start (MS) and forward-fill missing values
    df = df.resample('MS').ffill()
    
    # Calculate MoM and YoY CPI inflation rates
    df['MoM_Inflation'] = df['CPI'].pct_change(periods=1) * 100
    df['YoY_Inflation'] = df['CPI'].pct_change(periods=12) * 100
    
    # Calculate changes for exogenous variables to ensure stationarity
    if 'Crude_Oil' in df.columns:
        df['Crude_Oil_YoY_Change'] = df['Crude_Oil'].pct_change(periods=12) * 100
    if 'USD_INR' in df.columns:
        df['USD_INR_YoY_Change'] = df['USD_INR'].pct_change(periods=12) * 100
    
    # Keep Repo_Rate as a raw percentage rate (since it's already a rate)
    
    # ADF Test for stationarity on YoY Inflation
    yoy_clean = df['YoY_Inflation'].dropna()
    if len(yoy_clean) > 10:
        result = adfuller(yoy_clean)
        print("\n--- ADF Test on YoY Inflation ---")
        print(f"Test Statistic: {result[0]:.4f}")
        print(f"P-value: {result[1]:.4f}")
        if result[1] <= 0.05:
            print("Conclusion: YoY Inflation is stationary (Reject H0).")
        else:
            print("Conclusion: YoY Inflation is non-stationary (Fail to reject H0).")
            
    # Save preprocessed data
    df.to_csv(output_path)
    print(f"\nProcessed data saved to {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data()
