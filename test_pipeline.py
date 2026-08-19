import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import fetch_cpi_data
from src.preprocess import preprocess_data
from src.features import generate_features

def main():
    print("=== Step 1: Fetching Data ===")
    fetch_cpi_data()
    
    print("\n=== Step 2: Preprocessing ===")
    preprocess_data()
    
    print("\n=== Step 3: Feature Engineering ===")
    features = generate_features()
    print("\n[VERIFICATION] Final Features DataFrame (First 5 Rows):")
    print(features.head())
    print(f"\nTotal rows after removing NaNs: {len(features)}")
    
if __name__ == "__main__":
    main()
