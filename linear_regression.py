import numpy as np
from sklearn.linear_model import LinearRegression

def run_linear_regression(X, y):
    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    coef_ = model.coef_
    intercept_ = model.intercept_
    r2 = model.score(X, y)  # R^2

    return model, coef_, intercept_, r2