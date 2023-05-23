import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X, y)
y_predict = linear_regressor.predict(X)
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_predict, color="black")
plt.title("Linear regression")
plt.show()

polynomial = PolynomialFeatures(degree=2, include_bias=False)
X_poly = polynomial.fit_transform(X)
poly_regressor = linear_model.LinearRegression()
poly_regressor.fit(X_poly, y)
y_predict = poly_regressor.predict(X_poly)
print('X[0]', X[0])
print('X_poly', X_poly[0])
print("Polynomial regressor coefficient:", poly_regressor.coef_)
print("Polynomial regressor intercept:", poly_regressor.intercept_)
X_pd = pd.Series(X.flatten())
y_pred_pd = pd.Series(y_predict.flatten())
X_Sorted = X_pd.sort_values()
y_pred = np.array(y_pred_pd[X_Sorted.index])
X_arr = np.array(X_Sorted)

plt.figure()
plt.scatter(X, y)
plt.plot(X_arr, y_pred, color="black")
plt.title("Polynomial regression (degree 2)")
plt.show()
