# coding:utf-8
import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

import matplotlib.pyplot as plt


def access_test_samples(data, last_training_day=0.3, seed=1):
    days = data['day'].unique()
    np.random.seed(seed)
    np.random.shuffle(days)

    test_days = days[:int(len(days) * last_training_day)]
    data['is_test'] = data['day'].isin(test_days)
    # print(data['is_test'].unique())
    return data


def select_features(data):
    columns = data.columns[(data.dtypes == np.int64) | (data.dtypes == np.float64) | (data.dtypes == np.bool)].values
    return [feat for feat in columns if feat not in ['count', 'casual', 'registered'] and
            'log' not in feat]


def get_x_y(data, target_variable):
    features = select_features(data)
    x = data[features].values
    y = data[target_variable].values
    return x, y


def train_test_split(data, target_variable):
    train = data[data.is_test == False]
    test = data[data.is_test == True]

    x_train, y_train = get_x_y(train, target_variable)
    x_test, y_test = get_x_y(test, target_variable)
    return x_train, x_test, y_train, y_test


def fit_and_predict(data, model, target_variable):
    x_train, x_test, y_train, y_test = train_test_split(data, target_variable)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    return y_test, y_predict


def post_predict(y_predict):
    y_predict[y_predict < 0] = 0
    return y_predict


# Loss Function
def rmsle(y_true, y_predict, y_predict_only_positive=True):
    if y_predict_only_positive:
        y_predict = post_predict(y_predict)
    diff = np.log(y_predict + 1) - np.log(y_true + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)


# #### ####
def count_prediction(data, model, target_variable='count'):
    data = etl_datetime(access_test_samples(data))

    y_test, y_predict = fit_and_predict(data, model, target_variable)

    if target_variable == 'count_log':
        y_test = data[data['is_test'] == True]['count']
        y_predict = np.exp2(y_predict)
    return rmsle(y_test, y_predict)


def registered_casual_prediction(data, model):
    data = etl_datetime(access_test_samples(data))
    _, registered_predict = fit_and_predict(data, model, 'registered')
    _, casual_predict = fit_and_predict(data, model, 'casual')

    y_test = data[data['is_test'] == True]['count']
    y_predict = registered_predict + casual_predict
    return rmsle(y_test, y_predict)


def log_registered_casual_prediction(data, model):
    data = etl_datetime(access_test_samples(data))
    (_, registered_predict) = fit_and_predict(data, model, 'registered_log')
    (_, casual_predict) = fit_and_predict(data, model, 'casual_log')

    y_test = data[data['is_test'] == True]['count']
    y_predict = (np.exp2(registered_predict) - 1) + (np.exp2(casual_predict) - 1)

    return rmsle(y_test, y_predict)


# #### ####
def importance_features(model, data):
    imf = []
    booster = model.get_booster()
    f_score = booster.get_fscore()

    maps_name = dict([("f{0}".format(i), col) for i, col in enumerate(data.columns)])
    # print(maps_name)
    # print(f_score)

    for ft, score in f_score.items():
        imf.append({'feature': maps_name[ft], 'importance': score})
    imf_d = pd.DataFrame(imf)
    imf_d = imf_d.sort_values(by='importance', ascending=False).reset_index(drop=True)
    imf_d['importance'] /= imf_d['importance'].sum()
    return imf_d


def draw_importance_features(model, data):
    impdf = importance_features(model, data)
    print(impdf)
    return impdf.plot(kind='bar', title='Importance Features', figsize=(20, 8))


# #### #### feature engineering
def etl_datetime(df):
    df['year'] = df['datetime'].map(lambda x: x.year)
    df['month'] = df['datetime'].map(lambda x: x.month)

    df['hour'] = df['datetime'].map(lambda x: x.hour)
    df['minute'] = df['datetime'].map(lambda x: x.minute)
    df['day_of_week'] = df['datetime'].map(lambda x: x.dayofweek)
    df['weekend'] = df['datetime'].map(lambda x: x.dayofweek in [5, 6])

    return df


if __name__ == '__main__':
    d = pd.read_csv('bike.csv')  # 12 features
    d['datetime'] = pd.to_datetime(d['datetime'])
    d['day'] = d['datetime'].map(lambda x: x.day)  # day

    for max_depth in [2]:
        for n_estimators in [150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163]:
            params = {'max_depth': max_depth, 'n_estimators': n_estimators}
            m = xgb.XGBRegressor(**params)
            print(params, registered_casual_prediction(d, m))
    # draw_importance_features(m, d)
    # plt.show()
