# coding:utf-8

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# iris数据集

if __name__ == '__main__':
    iris = datasets.load_iris()  # dict = {'data':[[],...], 'target':[...]}
    gnb = GaussianNB()
    gnb.fit(iris.data, iris.target)
    y_predict = gnb.predict(iris.data)  # 数据集划分

    right_num = (iris.target == y_predict).sum()
    print("Total testing num :%d , naive bayes accuracy :%f" % (
        iris.data.shape[0], float(right_num) / iris.data.shape[0]))
