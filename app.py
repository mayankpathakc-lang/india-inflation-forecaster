import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Inflation Forecasting Dashboard", layout="wide")

st.title("📈 India Inflation Analyzer & Forecaster")
st.markdown("This dashboard tracks historical inflation (YoY vs MoM) and forecasts future inflation using SARIMA and XGBoost.")

@st.cache_data
def load_data():
    hist = pd.read_csv("data/processed/india_inflation.csv")
    hist['Date'] = pd.to_datetime(hist['Date'])
    
    metrics = None
    if os.path.exists("data/forecasts/metrics.csv"):
        metrics = pd.read_csv("data/forecasts/metrics.csv")
        
    forecasts = None
    if os.path.exists("data/forecasts/future_forecasts.csv"):
        forecasts = pd.read_csv("data/forecasts/future_forecasts.csv")
        forecasts['Date'] = pd.to_datetime(forecasts['Date'])
        
    return hist, metrics, forecasts

hist_df, metrics_df, forecast_df = load_data()

st.header("📊 Historical Trends")
if hist_df is not None and not hist_df.empty:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['YoY_Inflation'], mode='lines', name='YoY Inflation'))
    if 'MoM_Inflation' in hist_df.columns:
        fig_hist.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['MoM_Inflation'], mode='lines', name='MoM Inflation', opacity=0.7))
    st.plotly_chart(fig_hist, use_container_width=True)
    
    with st.expander("🌐 View Exogenous Macroeconomic Indicators"):
        exog_choice = st.selectbox("Select Indicator to Plot:", 
                                   ["Repo Rate (Policy Interest Rate)", "Brent Crude Oil Price (USD/Barrel)", "USD/INR Exchange Rate"])
        fig_exog = go.Figure()
        if "Repo Rate" in exog_choice and 'Repo_Rate' in hist_df.columns:
            fig_exog.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Repo_Rate'], mode='lines', name='Repo Rate', line=dict(color='purple')))
            fig_exog.update_layout(title="India Repo Rate (Policy Rate)", yaxis_title="Interest Rate (%)")
        elif "Brent Crude" in exog_choice and 'Crude_Oil' in hist_df.columns:
            fig_exog.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Crude_Oil'], mode='lines', name='Brent Crude', line=dict(color='brown')))
            fig_exog.update_layout(title="Brent Crude Oil Price", yaxis_title="USD per Barrel")
        elif 'USD_INR' in hist_df.columns:
            fig_exog.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['USD_INR'], mode='lines', name='USD/INR', line=dict(color='green')))
            fig_exog.update_layout(title="USD/INR Exchange Rate", yaxis_title="Rupees per USD")
        else:
            st.warning("Selected indicator data not available.")
        fig_exog.update_layout(xaxis_title="Date", hovermode="x unified")
        st.plotly_chart(fig_exog, use_container_width=True)
else:
    st.info("Historical data not available.")

st.header("📉 Model Validation")
if metrics_df is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen', subset=['MAE', 'RMSE', 'MAPE']))
    with col2:
        st.markdown("""
        **Metrics Explanation:**
        - **MAE:** Mean Absolute Error
        - **RMSE:** Root Mean Squared Error 
        - **MAPE:** Mean Absolute Percentage Error
        
        *Lower values are better.*
        """)
else:
    st.info("Metrics not found. Run the training pipeline first.")

st.header("🔮 Future Forecasts (Next 12 Months)")
if forecast_df is not None and hist_df is not None:
    model_choice = st.radio("Select Model for Forecast View:", ("SARIMA", "XGBoost"))
    
    fig_fc = go.Figure()
    # Plot last 3 years of history for context
    hist_recent = hist_df.tail(36)
    fig_fc.add_trace(go.Scatter(x=hist_recent['Date'], y=hist_recent['YoY_Inflation'], mode='lines', name='Historical YoY', line=dict(color='black')))
    
    if model_choice == "SARIMA":
        y = forecast_df['SARIMA_Forecast']
        y_lower = forecast_df['SARIMA_Lower']
        y_upper = forecast_df['SARIMA_Upper']
        color = 'blue'
    else:
        y = forecast_df['XGB_Forecast']
        y_lower = forecast_df['XGB_Lower']
        y_upper = forecast_df['XGB_Upper']
        color = 'red'
        
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([pd.Series([hist_recent['Date'].iloc[-1]]), forecast_df['Date']]),
        y=pd.concat([pd.Series([hist_recent['YoY_Inflation'].iloc[-1]]), y]),
        mode='lines', name=f'{model_choice} Forecast', line=dict(color=color)
    ))
    
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast_df['Date'], forecast_df['Date'][::-1]]),
        y=pd.concat([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor=f'rgba({0 if color=="blue" else 255},{0 if color=="red" else 0},255,0.2)' if color=='blue' else 'rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Confidence Interval'
    ))
    
    fig_fc.update_layout(title=f"{model_choice} 12-Month Forecast", xaxis_title="Date", yaxis_title="YoY Inflation (%)", hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)
else:
    st.info("Forecast data not available. Run the pipeline first.")
    
st.header("📚 Methodology")
st.markdown("""
### Data Preparation & Stationarity
The data represents the monthly CPI for India, processed into Year-over-Year (YoY) and Month-over-Month (MoM) inflation percentages. Stationarity checks (like ADF) typically confirm if differencing is required. 

### SARIMA Baseline
We use `pmdarima` for auto-ARIMA selection, fitting a Seasonal ARIMA model. It analytically estimates parameter uncertainty to construct confidence intervals.

### XGBoost with Recursive Forecasting
The ML approach builds lag features (e.g., lag_1, lag_3, lag_12) and rolling statistics (mean, std over 3, 6, 12 months). To forecast N steps ahead, the model makes a 1-step prediction, appends it to the sequence, recalculates the lag/rolling features, and repeats (Recursive Forecasting). Confidence intervals are estimated using residual standard error.
""")
