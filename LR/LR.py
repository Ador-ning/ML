# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data_set():
    data_matrix, label_matrix = [], []
    f = open('testSet2.txt')
    for line in f.readlines():
        line_list = line.strip().split()
        # 1.0  --> bias
        data_matrix.append([1.0, float(line_list[0]), float(line_list[1])])  # 样本数据
        label_matrix.append(int(line_list[2]))  # 样本标签
    return data_matrix, label_matrix


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def gradient_ascent(d_src, data_label):
    """
    # GD training
    :param d_src: 数据列表
    :param data_label: 数据对应类型标签
    :return:
    """

    data = np.matrix(d_src)  # 转矩阵
    label = np.matrix(data_label).transpose()  # T
    m, n = np.shape(data)  # 样本数据维度

    alpha = 0.001  # learning rate
    max_cycles = 500
    weights = np.ones((n, 1))  # parameters  (m, n) x (n, 1)

    for k in range(max_cycles):
        h = sigmoid(data * weights)
        error = label - h
        weights += alpha * data.T * error  # 更新 weights
    return weights


def s_gradient_ascent(d_src, data_label, num_iter=150):
    m, n = np.shape(d_src)
    weights = np.ones(n)

    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01

            # 随机挑选数据，进行参数更新
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(d_src[rand_index] * weights))
            error = data_label[rand_index] - h
            weights += alpha * error * d_src[rand_index]
    return weights


def plot_best_fit(weights):
    data, label = load_data_set()
    data_array = np.array(data)
    n = np.shape(data_array)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []

    # 按类别设置坐标点
    for i in range(n):
        if int(label[i]) == 1:
            x_cord1.append(data_array[i, 1])
            y_cord1.append(data_array[i, 2])
        else:
            x_cord2.append(data_array[i, 1])
            y_cord2.append(data_array[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    d, types = load_data_set()
    plot_best_fit(gradient_ascent(d, types))
