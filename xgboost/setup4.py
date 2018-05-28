# coding:utf-8

"""
Hyper parameter optimization
    1. Grid Search
    2. RandomizedSearch
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# 已有
from step3 import rmsle, fit_and_predict, train_test_split, access_test_samples
from step3 import etl_datetime

space = {
    'max_depth': hp.quniform('x_max_depth', 2, 20, 1),
    'n_estimators': hp.quniform('x_n_estimators', 100, 1000, 1),
    'subsample': hp.uniform('x_subsample', 0.8, 1),
    'colsample_bytree': hp.uniform('x_colsample_bytree', 0.1, 1),
    'learning_rate': hp.uniform('x_learning_rate', 0.01, 0.1),
    'reg_alpha': hp.uniform('x_reg_alpha', 0.1, 1)
}


def objective(space):
    model = xgb.XGBRegressor(
        max_depth=int(space['max_depth']),
        n_estimators=int(space['n_estimators']),  # 整数
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree'],
        learning_rate=space['learning_rate'],
        reg_alpha=space['reg_alpha'])

    X_train, X_test, y_train, y_test = train_test_split(d, 'count')
    eval_set = [(X_train, y_train), (X_test, y_test)]

    (_, registered_pred) = fit_and_predict(d, model, 'registered_log')
    (_, casual_pred) = fit_and_predict(d, model, 'casual_log')

    y_test = d[d.is_test == True]['count']
    y_pred = (np.exp2(registered_pred) - 1) + (np.exp2(casual_pred) - 1)

    score = rmsle(y_test, y_pred)
    print("SCORE:", score)

    return {'loss': score, 'status': STATUS_OK}


if __name__ == '__main__':
    d = pd.read_csv('bike.csv')
    d['datetime'] = pd.to_datetime(d['datetime'])

    d['day'] = d['datetime'].map(lambda x: x.day)
    d = etl_datetime(access_test_samples(d))  # 解析时间 (10886, 19)

    d['{0}_log'.format('count')] = d['count'].map(lambda x: np.log2(x))  # 对d['count'] 取log
    for name in ['registered', 'casual']:
        d['{0}_log'.format(name)] = d[name].map(lambda x: np.log2(x + 1))  # 对d['registered'] d['casual'] 取log
    # print(d)  (10886, 22)
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=15,
                trials=trials)

    print(best)
