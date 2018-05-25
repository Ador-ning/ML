# coding:utf-8
import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt


def ground_truth(x):
    """ Function to approximate """
    return x * np.sin(x) + 2 * np.sin(2 * x) + np.sin(3 * x)


def generate_data(n_samples=200):
    """ Generate training and testing data """
    np.random.seed(15)
    x = np.random.uniform(0, 10, size=n_samples)[:, np.newaxis]
    y = ground_truth(x.ravel()) + np.random.normal(scale=2, size=n_samples)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    return x_train, x_test, y_train, y_test


def plot_data(alpha=0.4, s=20):
    fig = plt.figure(dpi=100)
    gt = plt.plot(x_plot, ground_truth(x_plot), alpha=alpha, label="ground truth")

    # plot training and testing data
    plt.scatter(X_train, Y_train, s=s, alpha=alpha)
    plt.scatter(X_test, Y_test, s=s, alpha=alpha, edgecolors='red')
    plt.xlim((0, 10))
    plt.ylabel('Y')
    plt.xlabel('X')


def rt_hyper_param():
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X_train, Y_train)
    X_predict1 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict1, label="RT max_depth=1", color='r', alpha=0.9, linewidth=1)

    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X_train, Y_train)
    X_predict2 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict2, label="RT max_depth=3", color='y', alpha=0.7, linewidth=1)

    model = DecisionTreeRegressor(max_depth=7)
    model.fit(X_train, Y_train)
    X_predict3 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict3, label="RT max_depth=10", color='b', alpha=0.5, linewidth=1)
    plt.legend(loc='upper left')

    plt.show()


def rf_hyper_params():
    # The number of trees in the forest ---> n_estimators   default --> 10
    model = RandomForestRegressor(n_estimators=1, max_depth=1)
    model.fit(X_train, Y_train)
    X_predict1 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict1, label="RF n_estimators=1 max_depth=1", color='r', alpha=0.9, linewidth=1)

    model = RandomForestRegressor(n_estimators=1, max_depth=3)
    model.fit(X_train, Y_train)
    X_predict2 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict2, label="RF n_estimators=1 max_depth=3", color='y', alpha=0.7, linewidth=1)

    model = RandomForestRegressor(n_estimators=5, max_depth=1)
    model.fit(X_train, Y_train)
    X_predict3 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict3, label="RF n_estimators=5 max_depth=1", color='g', alpha=0.5, linewidth=1)
    plt.legend(loc='upper left')

    plt.show()


def xgb_hyper_params():
    model = xgb.XGBRegressor(n_estimators=1, max_depth=5)
    model.fit(X_train, Y_train)
    X_predict1 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict1, label="xgb n_estimators=1 max_depth=5", color='r', alpha=0.9, linewidth=1)

    model = xgb.XGBRegressor(n_estimators=10, max_depth=5)
    model.fit(X_train, Y_train)
    X_predict2 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict2, label="xgb n_estimators=1 max_depth=5", color='y', alpha=0.7, linewidth=1)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, Y_train)
    X_predict3 = model.predict(x_plot[:, np.newaxis])
    plt.plot(x_plot, X_predict3, label="xgb n_estimators=100 max_depth=5", color='g', alpha=0.5, linewidth=1)
    plt.legend(loc='upper left')

    plt.show()


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = generate_data(n_samples=100)
    x_plot = np.linspace(0, 10, 500)
    plot_data()
    # rt_hyper_param()
    # rf_hyper_params()
    xgb_hyper_params()
    annotation_kw = {'xycoords': 'data', 'textcoords': 'data',
                     'arrowprops': {'arrowstyle': '->', 'connectionstyle': 'arc'}}
