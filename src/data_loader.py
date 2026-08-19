import os
import pandas as pd
import requests
from glob import glob

def fetch_series(series_id, label, output_path, api_key=None):
    df = None
    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            series = fred.get_series(series_id)
            df = pd.DataFrame(series, columns=[label])
            df.index.name = 'Date'
            df.reset_index(inplace=True)
            print(f"Fetched {label} via fredapi.")
        except Exception as e:
            print(f"fredapi failed for {label}: {e}. Falling back to URL download.")
    
    if df is None:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        print(f"Fetching {label} via direct URL: {url}")
        try:
            df = pd.read_csv(url)
            df.columns = ['Date', 'Value']
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df.rename(columns={'Value': label}, inplace=True)
            print(f"Fetched {label} via direct URL.")
        except Exception as e:
            print(f"Failed to fetch {label} via URL: {e}")
            return None

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Save raw CSV
    df.to_csv(output_path, index=False)
    return df

def fetch_all_data(output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    api_key = os.environ.get("FRED_API_KEY")
    
    series_dict = {
        'INDCPIALLMINMEI': 'CPI',
        'IRSTCB01INM156N': 'Repo_Rate',
        'POILBREUSDM': 'Crude_Oil',
        'EXINUS': 'USD_INR'
    }
    
    dfs = []
    for series_id, label in series_dict.items():
        out_path = os.path.join(output_dir, f"{label.lower()}.csv")
        df = fetch_series(series_id, label, out_path, api_key)
        if df is not None:
            dfs.append(df)
            
    if not dfs:
        raise ValueError("Could not fetch any data series from FRED.")
        
    # Merge all FRED series on Date
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='outer')
        
    # Merge with RBI manual data if present
    rbi_files = glob(os.path.join(output_dir, "rbi_manual/*.csv"))
    for f in rbi_files:
        try:
            rbi_df = pd.read_csv(f)
            if 'Date' in rbi_df.columns:
                rbi_df['Date'] = pd.to_datetime(rbi_df['Date'])
                merged_df = pd.merge(merged_df, rbi_df, on='Date', how='outer')
                print(f"Merged RBI manual data from {f}")
            else:
                print(f"No 'Date' column found in {f}. Skipping.")
        except Exception as e:
            print(f"Failed to merge {f}: {e}")
            
    merged_df.sort_values('Date', inplace=True)
    
    combined_raw_path = os.path.join(output_dir, "india_cpi.csv") # keep the same filename for compatibility or update it
    merged_df.to_csv(combined_raw_path, index=False)
    print(f"Combined raw data saved to {combined_raw_path}")
    return merged_df

if __name__ == "__main__":
    fetch_all_data()
