import pmdarima as pm
import pandas as pd
import numpy as np

def train_sarima(train_series, X=None):
    """
    Fits a SARIMA/SARIMAX model on processed YoY_Inflation.
    Supports optional exogenous features X.
    """
    model = pm.auto_arima(
        train_series,
        X=X,
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        m=12,
        start_P=0, seasonal=True,
        d=None, D=1, trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    return model

def forecast_sarima(model, n_steps=12, X=None, alpha=0.05):
    """
    Forecasts next N steps with analytical confidence intervals.
    Supports optional exogenous features X.
    """
    forecasts, conf_int = model.predict(n_periods=n_steps, X=X, return_conf_int=True, alpha=alpha)
    return forecasts, conf_int
