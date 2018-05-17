# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data = np.matrix(
        [[1.0, 2.1],
         [2.0, 1.1],
         [1.3, 1.0],
         [1.0, 1.0],
         [2.0, 1.2]])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data, labels


def stump_classify(data, dimen, thresh_val, thresh_in):
    """
    # 通过阈值比较对数据进行分类的
    :param data: 数据
    :param dimen: 特征列
    :param thresh_val: 特征比较值
    :param thresh_in: 符号
    :return: np.array
    """
    ret_array = np.ones((np.shape(data)[0], 1))  # 299x1

    if thresh_in == 'lt':
        ret_array[data[:, dimen] <= thresh_val] = -1.0  # 第dimen列特征
    else:
        print(type(data[:, dimen]))
        print(type(thresh_val))
        ret_array[data[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data, labels, d):
    """
    # 遍历stump_classify()函数所有的可能输入值，并找到数据机上最佳的单层决策树
    :param data:
    :param labels:
    :param d: 数据权重向量
    :return: 最优分类器模型，错误率，模型预测的结果
    """
    data_matrix = np.matrix(data).astype(float)
    labels_matrix = np.matrix(labels).T
    # labels_matrix = int(float(labels_matrix))

    m, n = np.shape(data_matrix)  # (m, n)
    num_steps = 10.0  # 用于在特征的所有可能值上进行遍历
    best_stump = {}  # 存储给定权重向量d时，所得到的最佳单层决策树的相关信息
    best_class_est = np.matrix(np.zeros((m, 1)))
    min_error = np.inf

    for i in range(n):  # 遍历所有数据集的特征
        # print(data_matrix[:, i])
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps  # 步长

        for j in range(-1, int(num_steps) + 1):  #

            for equal in ["lt", 'gt']:  # 切换方向
                thresh_val = range_min + float(j) * step_size  # 特征比较值的调整

                # 预测结果
                predicted_val = stump_classify(data_matrix, i, thresh_val, equal)  # 不同 比较值 / 左右方向
                error = np.matrix(np.ones((m, 1)))
                error[predicted_val == labels_matrix] = 0  # 分类错误的为 1

                # (1, m)x(m, 1)
                weight_error = d.T * error  # 利用数据权重，来评价分类器
                # print("split:dim %d, thresh %.2f, thresh equal: %s, the weighted error is % .3f" %
                #      (i, thresh_val, equal, weight_error))

                if weight_error < min_error:  # 保存最优分类器
                    min_error = weight_error
                    best_class_est = predicted_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['eq'] = equal
    return best_stump, min_error, best_class_est


def ada_boost_train_ds(data_array, labels, num_iter=40):
    """
    :param data_array:
    :param labels:
    :param num_iter: 迭代次数
    :return: 弱分类器集合，预测分类的结果
    """
    weak_class = []
    m = np.shape(data_array)[0]  # 数据数目
    d = np.matrix(np.ones((m, 1)) / m)  # 数据初始系数
    agg_class_est = np.matrix(np.zeros((m, 1)))

    for i in range(num_iter):
        best_stump, error, class_est = build_stump(data_array, labels, d)
        # print("d:", d.T)  # 数据系数
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 模型系数alpha
        best_stump['alpha'] = alpha

        weak_class.append(best_stump)
        # print("class_est:", class_est.T)
        expon = np.multiply(-1 * alpha * np.matrix(labels).T, class_est)
        d = np.multiply(d, np.exp(expon))  # 更新数据参数
        d = d / d.sum()

        agg_class_est += alpha * class_est
        # print("agg_class_est:", agg_class_est.T)
        # 模型融合 误差
        agg_errors = np.multiply(np.sign(agg_class_est) != np.matrix(labels).T, np.ones((m, 1)))
        agg_rate = agg_errors.sum() / m
        print(" error rate :", agg_rate)
        if agg_rate == 0:
            break
    return weak_class, agg_class_est


def load_data_set(filename):
    """
    :param filename:
    :return:
    """
    number_feature = len(open(filename).readline().split('\t'))
    data_array = []
    label_array = []

    f = open(filename)
    for line in f.readlines():
        line_list = line.strip().split('\t')
        data_array.append(line_list[:number_feature - 1])
        label_array.append(int(float(line_list[-1])))
        # print(line_list[:number_feature - 1])
        # print(line_list[-1])
    return data_array, label_array


def ada_classify(data, classifier_array):
    """
    # 通过训练得到的弱分类器的集合进行预测
    :param data: 测试数据
    :param classifier_array: 分类器列表
    :return: 预测结果
    """
    data_inner = np.matrix(data)
    m = np.shape(data_inner)[0]
    agg_class_est = np.matrix(np.zeros((m, 1)))

    for i in range(len(classifier_array)):
        class_est = stump_classify(data_inner, classifier_array[i]['dim'], classifier_array[i]['thresh'],
                                   classifier_array[i]['eq'])  # 分类
        agg_class_est += classifier_array[i]['alpha'] * class_est  # 加权表决
        # print(agg_class_est)
    return np.sign(agg_class_est)


def plot_roc(predict_weights, class_label):
    """
    # 画 ROC 曲线图，并计算 AUC 的面积大小
    :param predict_weights: 最终预测结果的权重值
    :param class_label: 原始数据的分类结果集
    :return:
    """
    pass


def test():
    data_train, labels_train = load_data_set('./horseColicTraining2.txt')
    print('training data:', np.shape(data_train))
    weak_class_array, agg_class_est = ada_boost_train_ds(data_train, labels_train)
    print("weak_class_array:", weak_class_array)
    # print("agg_class_est", agg_class_est.T)  # 训练
    plot_roc(agg_class_est, labels_train)

    print('Test:')
    data_test, labels_test = load_data_set('./horseColicTest2.txt')
    test_len = np.shape(data_test)[0]
    print('testing data:', np.shape(data_test))

    predict = ada_classify(data_test, weak_class_array)
    error = np.matrix(np.zeros((test_len, 1)))
    print("Rate:", error[predict != np.matrix(labels_test).T].sum() / test_len)


if __name__ == '__main__':
    test()
