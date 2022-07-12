import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

df = df = pd.read_excel('./data/Dat9-6.xls', skiprows=[0])


def plotDispersion(df):
    # Plot dispersion of assets and debt
    plt.plot(
        df[df.columns[0]].values,
        np.arange(0, df[df.columns[0]].values.shape[0]),
        '--o',
        label=df[df.columns[0]].name)

    plt.plot(
        df[df.columns[1]].values,
        np.arange(0, df[df.columns[1]].values.shape[0]),
        '--o',
        label=df[df.columns[1]].name)

    plt.title('Asset and Debt dispersion chart')
    plt.legend()
    plt.show()


def plotRegression(df, degree):
    x = df[df.columns[0]].values
    y = df[df.columns[1]].values

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(x.reshape(-1, 1))

    # Create linear regression model
    reg = linear_model.LinearRegression()
    reg.fit(poly_features, y)

    # Predict
    pred = reg.predict(poly_features)

    # Score
    score = reg.score(poly_features, y)

    # Plot
    plt.plot(x, pred, '--o', label='Prediction')
    plt.plot(x, y, '--o', label='Real data')

    plt.suptitle(
        'Linear regression {}º - Score R²: {:.4f}'.format(degree, score))

    # Add equation of regression for degree = n
    plt.title(
        'y = ' + ' + '.join([f'{reg.coef_[i]:.3f}x^{i}' for i in range(degree)]))

    plt.legend()
    plt.show()


def plotEstimate(df, degree):
    # Split data
    x = df[df.columns[0]].values
    y = df[df.columns[1]].values

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(x.reshape(-1, 1))

    # Create linear regression model
    reg = linear_model.LinearRegression()
    reg.fit(poly_features, y)

    # Predict
    pred = reg.predict(poly_features)

    # Plot area with 95% confidence interval
    plt.fill_between(x, pred - 1.95 * np.sqrt(y),
                     pred + 1.95 * np.sqrt(y), alpha=0.5)
    plt.plot(x, y, '--o', label='Real data')
    plt.title("DEBT CONFIDENCE INTERVAL")
    plt.legend()
    plt.show()


print("Correlation bettwen Assets and Debt: {}".format(
    df[df.columns[0]].corr(df[df.columns[1]])))

# Base description
print(df.describe())

# Dispersion
plotDispersion(df)

# Regression with more than 0.95 score
# Therefore, it is possible to use linear regression for this case
plotRegression(df, degree=8)

plotEstimate(df, degree=8)
