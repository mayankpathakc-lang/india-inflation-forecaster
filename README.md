# India Inflation Forecasting

## Project Goal
This project aims to accurately forecast India's Inflation Rate (Year-over-Year percentage change in the Consumer Price Index) using modern time-series forecasting techniques and machine learning. By predicting the inflation rate, we can gain insights into future economic conditions, aiding policymakers, businesses, and individuals in decision-making.

## Architecture & File Structure
```text
.
├── data/
│   ├── raw/            # Raw data downloaded from APIs or CSVs
│   └── processed/      # Cleaned and feature-engineered datasets
├── notebooks/          # Jupyter notebooks for exploratory data analysis (EDA)
├── src/                # Core source code
│   ├── data_fetch.py   # Handles data ingestion from FRED API/CSVs
│   ├── features.py     # Feature engineering (lags, rolling windows, cyclical)
│   ├── train.py        # Model training and evaluation
│   └── inference.py    # Forecasting future values
├── app.py              # Streamlit web application
├── forecast.py         # CLI tool to run forecasts
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
└── README.md           # Project documentation
```
*(Note: adjust actual file structure based on your specific implementation)*

## Data Sourcing
The primary data source for the Consumer Price Index (CPI) is the **FRED (Federal Reserve Economic Data) API**. The project uses the FRED API to automatically fetch the latest data for India's CPI.
- **Fallback Mechanism:** In case the API is unavailable or rate-limited, the system falls back to reading a locally stored CSV file (`data/raw/cpi_india.csv`), ensuring robust execution.
- **RBI DBIE Data:** For users wishing to incorporate more granular or localized data from the Reserve Bank of India (RBI) Database on Indian Economy (DBIE), the data pipeline supports manual merging of these datasets to enrich the primary CPI data.

### CPI vs. Inflation Rate
It is important to distinguish between the CPI Price Index and the actual Inflation Rate. 
- **CPI (Consumer Price Index):** A measure that examines the weighted average of prices of a basket of consumer goods and services. It is an absolute index number (e.g., 150.5).
- **Inflation Rate:** The percentage change in the CPI over a specific period, usually calculated Year-over-Year (YoY). Forecasting the percentage change (Inflation Rate) rather than the raw CPI is standard practice because the inflation rate is typically stationary or closer to stationary, whereas raw CPI tends to have a strong non-stationary upward trend, making it harder for models to learn meaningful patterns.

## Core Time-Series Concepts
To build robust forecasting models, we utilize several key time-series concepts:

### Stationarity & ADF Test
A time-series is stationary if its statistical properties (mean, variance) remain constant over time. Most forecasting models assume or perform better with stationary data. 
- **Differencing:** We transform non-stationary data into stationary data by calculating the difference between consecutive observations (e.g., YoY inflation).
- **Augmented Dickey-Fuller (ADF) Test:** A statistical test used to check whether a time-series is stationary. We use this to validate our data transformations.

### Lags and Rolling Windows Features
- **Lag Features:** Past values of the target variable (e.g., inflation rate 1 month ago, 3 months ago) are used as predictors for the current value.
- **Rolling Windows:** Statistical aggregations over a recent period (e.g., 3-month rolling mean, 6-month rolling standard deviation) capture short-term trends and volatility.

### Cyclical Month Encoding
Time is cyclical (January follows December). Representing months as raw integers (1 to 12) misleads models into thinking December (12) is "farther" from January (1) than it is from November (11). We encode months using **sine and cosine transformations** to correctly represent the cyclical nature of time.

## Evaluation Strategy: Chronological Split vs. Random Shuffling
In traditional machine learning, data is often randomly shuffled and split into training and testing sets. **This approach is invalid for time-series forecasting.**
- **Data Leakage:** Randomly splitting time-series data causes "leakage," where the model learns from future information to predict the past.
- **Chronological Split / Walk-Forward Backtesting:** We strictly split the data chronologically (e.g., train on 2010-2020, test on 2021-2023). For robust evaluation, we use expanding or rolling window walk-forward validation, simulating how the model would perform in the real world as new data arrives.

## Modeling Choices
We employ a two-tiered modeling approach:

1. **SARIMA Baseline (auto_arima):**
   - **Seasonal AutoRegressive Integrated Moving Average:** A classic statistical model that explicitly models trend and seasonality. We use `pmdarima`'s `auto_arima` to automatically discover the optimal (p, d, q) and seasonal (P, D, Q, s) parameters. It serves as our strong baseline.

2. **XGBoost (Machine Learning):**
   - **Extreme Gradient Boosting:** A powerful tree-based ensemble method. We feed it our engineered features (lags, rolling stats, cyclical encoding). 
   - **Recursive Forecasting:** Because XGBoost does not natively forecast multiple steps ahead like SARIMA, we use recursive forecasting: we predict step `t+1`, append the prediction to the dataset, recalculate lag/rolling features, and then predict step `t+2`.

## Evaluation Metrics
We evaluate model performance using standard regression metrics:
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual values. It is easy to interpret (e.g., "The model is off by 0.5% on average").
- **RMSE (Root Mean Squared Error):** The square root of the average squared differences. It heavily penalizes larger errors, useful when large forecasting misses are particularly costly.
- **MAPE (Mean Absolute Percentage Error):** The average absolute percentage difference. It provides a relative measure of error, making it easier to compare performance across different datasets or baselines.

## Project Setup & Running Guide

### 1. Requirements and Setup
Clone the repository and install the dependencies:
```bash
git clone <repository-url>
cd <project-dir>
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your FRED API key (if applicable):
```env
FRED_API_KEY=your_api_key_here
```

### 3. Running the Forecast Runner
You can execute the pipeline and generate forecasts via the CLI:
```bash
python forecast.py
```

### 4. Launching the Streamlit App
To view the interactive dashboard with data visualizations and forecast comparisons:
```bash
streamlit run app.py
```

## Final Benchmark Metrics

Below are the performance metrics evaluated on the chronological validation set (last 20% of historical data):

| Model | MAE | RMSE | MAPE |
| :--- | :---: | :---: | :---: |
| **SARIMAX** (Baseline with Exog) | 2.38 | 2.75 | 39.16% |
| **XGBoost** (Multivariate Recursive) | **1.28** | **1.74** | **22.10%** |

> [!NOTE]
> **Understanding the High MAPE**: Mean Absolute Percentage Error (MAPE) is exceptionally sensitive when actual values are close to zero. Since India's YoY inflation rate occasionally drops near 0% (or exhibits low volatility at certain intervals), minor absolute prediction errors (e.g., predicting 1.5% when actual is 0.2%) result in extremely high percentage errors, driving up the average MAPE. In such cases, MAE (Mean Absolute Error) is a more stable indicator of forecast performance.
