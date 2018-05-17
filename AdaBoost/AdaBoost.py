# coding:utf-8
import numpy as np


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
    :param data:
    :param dimen:
    :param thresh_val: 分类阈值
    :param thresh_in:
    :return:
    """
    ret_array = np.ones((np.shape(data)[0], 1))

    if thresh_in == 'It':
        ret_array[data[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data, labels, d):
    """
    # 遍历stump_classify()函数所有的可能输入值，并找到数据机上最佳的单层决策树
    :param data:
    :param labels:
    :param d: 数据权重向量
    :return:
    """
    data_matrix = np.matrix(data)
    labels_matrix = np.matrix(labels).T

    m, n = np.shape(data_matrix)  # (m, n)
    num_steps = 10.0  # 用于在特征的所有可能值上进行遍历
    best_stump = {}  # 存储给定权重向量d时，所得到的最佳单层决策树的相关信息
    best_class_est = np.matrix(np.zeros((m, 1)))
    min_error = np.inf

    for i in range(n):  # 遍历所有数据集的特征
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps  # 步长

        for j in range(-1, int(num_steps) + 1):  #

            for equal in ["lt", 'gt']:  # 切换 > / <
                thresh_val = range_min + float(j) * step_size

                # 预测结果
                predicted_val = stump_classify(data_matrix, i, thresh_val, equal)

                error = np.matrix(np.ones((m, 1)))
                error[predicted_val == labels_matrix] = 0

                weight_error = d.T * error  # 利用数据权重，来评价分类器
                print("split:dim %d, thresh %.2f, thresh equal: %s, the weighted error is % .3f" %
                      (i, thresh_val, equal, weight_error))
                if weight_error < min_error:
                    min_error = weight_error
                    best_class_est = predicted_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['eq'] = equal
    return best_stump, min_error, best_class_est


# 有问题
def ada_boost_train(data_array, labels, num_iter=40):
    weak_class = []
    m = np.shape(data_array)[0]  # 数据数目
    d = np.matrix(np.ones((m, 1)) / m)  # 数据初始系数
    agg_class_est = np.matrix(np.zeros((m, 1)))

    for i in range(num_iter):
        best_stump, error, class_est = build_stump(data_array, labels, d)
        print("d:", d.T)  # 数据系数

        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))  # 模型系数alpha

        best_stump['alpha'] = alpha
        weak_class.append(best_stump)
        print("class_est:", class_est.T)

        expon = np.multiply(-1 * alpha * np.matrix(labels).T, class_est)
        d = np.multiply(d, np.exp(expon))  # 跟新数据参数
        d = d / d.sum()

        agg_class_est += alpha * class_est
        print("agg_class_est:", agg_class_est.T)

        # 模型融合 误差
        agg_errors = np.multiply(np.sign(agg_class_est) != np.matrix(labels).T, np.ones((m, 1)))
        agg_rate = agg_errors.sum() / m
        print(" error rate :", agg_rate)
        if agg_rate == 0:
            break
    return weak_class


if __name__ == '__main__':
    # d_ = np.matrix(np.ones((5, 1)) / 5)
    data_all, label = load_data()
    print(ada_boost_train(data_all, label, num_iter=9))
