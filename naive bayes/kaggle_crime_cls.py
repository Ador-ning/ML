# coding:utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import time

if __name__ == '__main__':
    train = pd.read_csv('', parse_dates=['Dates'])
    test = pd.read_csv('', parse_dates=['Dates'])  # 解析 CSV

    # 用 Label Encoder对不同的犯罪类型进行标号
    leCrime = preprocessing.LabelEncoder()
    crime = leCrime.fit_transform(train.Category)

    # 因子化星期几， 街区，小时等特征
    days = pd.get_dummies(train.DayofWeek)
    district = pd.get_dummies(train.PdDistrict)
    hour = train.Dates.dt.hour
    hour = pd.get_dummies(hour)

    # 组合特征
    trainData = pd.concat([hour, days, district], axis=1)
    trainData['crime'] = crime

    # 测试集
    days = pd.get_dummies(test.DayOfWeek)
    district = pd.get_dummies(test.PdDistrict)
    hour = test.Dates.dt.hour
    hour = pd.get_dummies(hour)
    testData = pd.concat([hour, days, district], axis=1)

    # 星期几 和 街区 作为分类器输入特征
    features = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL',
                'TENDERLOIN']

    hourFea = [x for x in range(24)]
    features = features + hourFea

    # 分割训练集(3/5) 和 测试集(2/5)
    training, validation = train_test_split(trainData, train_size=.60)

    # 朴素贝叶斯建模， 计算 log_loss
    model = BernoulliNB()
    nbStart = time.time()
    model.fit(training[features], training['crime'])  # 训练  #### test data 没有用
    nbTrainTime = time.time() - nbStart
    predict = np.array(model.predict_log_proba(validation[features]))  # validation
    print("naive bayes 建模时间 %f s." % nbTrainTime)
    print("naive bayes log 损失为 %f." % (log_loss(validation['crime'], predict)))

    # 逻辑回归建模，计算log_loss
    model = LogisticRegression(C=.01)
    lrStart = time.time()
    model.fit(training[features], training['crime'])
    lrCostTime = time.time() - lrStart
    predicted = np.array(model.predict_proba(validation[features]))
    log_loss(validation['crime'], predicted)
    print("逻辑回归建模耗时 %f 秒" % lrCostTime)
    print("逻辑回归log损失为 %f" % (log_loss(validation['crime'], predicted)))
