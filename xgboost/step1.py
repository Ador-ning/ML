# coding:utf-8

import xgboost as xgb
# Load and return the boston house-prices data-set (regression)
from sklearn.datasets import load_boston, load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, f1_score, auc
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid", palette="husl")


def test_regression():
    boston = load_boston()  # (506, 13) np.array
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)
    model = xgb.XGBRegressor()  # 超参数  -- 默认
    model.fit(X_train, y_train)

    print(np.shape(model.apply(X_test)))
    y_predict = model.predict(X_test)
    print(r2_score(y_test, y_predict))


def test_classify():
    iris = load_iris()  # dict: {data: [[], ...], target: [[], ...]}
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
    model = xgb.XGBClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print(f1_score(y_test, y_predict, average="micro"))


if __name__ == '__main__':
    test_classify()
    pass
