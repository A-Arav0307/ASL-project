import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper function: Calculate R² score
def r2_score(y_true, y_pred):
    """Calculate R² (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Helper function: Calculate MSE
def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

# Helper function: Calculate MAE
def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

# Helper function: Linear Regression using numpy
def linear_regression_fit(X, y):
    """
    Fit linear regression using least squares
    Returns: slope, intercept
    """
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # Calculate slope
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    slope = numerator / denominator
    
    # Calculate intercept
    intercept = y_mean - slope * X_mean
    
    return slope, intercept

def linear_regression_predict(X, slope, intercept):
    """Make predictions using linear regression parameters"""
    return slope * X + intercept

# Helper function for regression analysis
def regression_analysis(y_true, y_pred, metric_name="Metric"):
    """
    Comprehensive regression analysis helper function
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero in MAPE
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    residuals = y_true - y_pred
    
    print(f"{'='*50}")
    print(f"Regression Analysis: {metric_name}")
    print(f"{'='*50}")
    print(f"R² Score:                 {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root MSE (RMSE):          {rmse:.6f}")
    print(f"Mean Absolute Error:      {mae:.6f}")
    print(f"MAPE:                     {mape:.2f}%")
    print(f"{'='*50}\n")
    
    return {
        'r2': r2, 'mse': mse, 'rmse': rmse, 
        'mae': mae, 'mape': mape, 'residuals': residuals
    }