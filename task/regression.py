import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def extend_predictor(x):
    ones_vector = np.ones((x.shape[0], 1))
    extendido = np.concatenate((ones_vector, x), axis=1)
    return extendido


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X, y):
        X = X.values
        y = y.values.reshape(-1, 1)
        if self.fit_intercept:
            X = extend_predictor(X)
        pseudo = np.linalg.inv(X.T @ X) @ X.T
        beta = (pseudo @ y).flatten()
        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]
        else:
            self.coefficient = beta

    def predict(self, X):
        beta = self.coefficient
        X = X.values
        if self.fit_intercept:
            beta = np.hstack((self.intercept, beta))
            X = extend_predictor(X)
        prediccion = X @ beta
        return prediccion

    def r2_score(self, y, yhat):
        y = y.values
        num = ((y - yhat) ** 2).sum()
        den = ((y - y.mean()) ** 2).sum()
        r2 = 1 - num / den
        return r2

    def rmse(self, y, yhat):
        y = y.values
        squared_errors = (y - yhat) ** 2
        rmse = np.sqrt(squared_errors.mean())
        return rmse


df = pd.read_csv('data.csv')
X = df.drop('y', axis=1)
y = df['y']
model = CustomLinearRegression(fit_intercept=True)
regSci = LinearRegression(fit_intercept=True)

model.fit(X, y)
regSci.fit(X, y)

yhat_m = model.predict(X)
yhat_r = regSci.predict(X)

r2_m = model.r2_score(y, yhat_m)
r2_r = r2_score(y, yhat_r)

rmse_m = model.rmse(y, yhat_m)
rmse_r = mean_squared_error(y, yhat_r, squared=False)

results = {'Intercept': regSci.intercept_ - model.intercept,
           'Coefficient': regSci.coef_ - model.coefficient,
          'R2': r2_r - r2_m,
                'RMSE': rmse_r - rmse_m
}
print(results)
