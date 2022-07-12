import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_excel('./data/Dat9-12.xls', skiprows=[0])


def view(df):
    print(df.describe())

    # Plot dispersion of Risk and Return columns
    plt.plot(df['Risk'],  label='Risk', marker='x')
    plt.plot(df['Return'], label='Return', marker='x')
    plt.plot(df['PE Ratio'], label='PE Ratio', marker='x')
    plt.legend()
    plt.show()

# The best attribute to adjust is the return
# view(df)


def linearMultivareatRegression(df, degree=3):
    y = df['Return']

    X2 = df['Risk']
    X1 = df['PE Ratio']

    poly = PolynomialFeatures(degree=degree)
    X = poly.fit_transform(np.array([X1, X2]).T)

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    pred = reg.predict(X)
    score = reg.score(X, y)

    # Plot the prediction of return based on the risk and the PE Ratio
    plt.plot(X2, y, 'o', label='Risk')
    plt.plot(X2, pred, 'o', label='Prediction')
    plt.suptitle(
        'Linear regression {}º - Score R²: {:.4f}'.format(degree, score))
    plt.legend()
    plt.show()

# Score R2 represents a great adjustment, without overfit
linearMultivareatRegression(df, 2)
# Already with a degree of 4 the score is exactly 1, with overfit
linearMultivareatRegression(df, 4)
