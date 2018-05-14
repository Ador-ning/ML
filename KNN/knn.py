# coding:utf-8

import numpy as np
import operator


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, data_set, labels, k):
    """
    :param inX: 用于分类的输入向量
    :param data_set: 输入的训练样本集
    :param labels: 训练样本标签
    :param k: 选择最近邻居的数目
    :return: 类别
    """
    data_set_size = data_set.shape[0]

    # 计算距离
    diff_mat = np.tile(inX, (data_set_size, 1)) - data_set  # 减
    sq_diff_math = diff_mat ** 2  # 平方
    sq_distances = sq_diff_math.sum(axis=1)  # 求和
    distances = sq_distances ** 0.5  # 开根号

    # 升排序
    sorted_dis_indicies = distances.argsort()

    # 选择距离小的 k 个点
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dis_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # 排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    number_lines = len(lines)
    return_matrix = np.zeros((number_lines, 3))  # 创建返回矩阵

    class_label = []  # 分类结果 label
    index = 0
    for line in lines:
        list_line = line.strip().split('\t')
        return_matrix[index:] = list_line[:3]
        class_label.append(list_line[-1])
        index += 1
    return return_matrix, class_label


def auto_norm(data):
    """
    # 对数据进行归一化
    :param data:
    :return:
    """
    min_val = data.min(0)
    max_val = data.max(0)
    ranges = max_val - min_val
    norm_data = np.zeros(shape=data.shape)

    m = data.shape[0]
    norm_data = data - np.tile(min_val, (m, 1))  #
    norm_data = norm_data / np.tile(ranges, (m, 1))  #
    return norm_data, ranges, min_val


def date_classify_test():
    ratio = 0.3
    data_matrix, labels = file2matrix('datingTestSet.txt')
    norm_data, ranges, min_values = auto_norm(data_matrix)
    m = norm_data.shape[0]
    number_test = int(m * ratio)
    error_count = 0
    for i in range(number_test):
        classify_result = classify0(norm_data[i, :], norm_data[number_test:m, :],
                                    labels[number_test:m], 3)
        if classify_result != labels[i]:
            error_count += 1
        print('classify type %s, real type %s' % (classify_result, labels[i]))
    print('total error ratio %f.' % (float(error_count) / float(m)))


if __name__ == '__main__':
    # group, labels = create_data_set()
    # print(classify0([0, 0], group, labels, 3))
    # data_matrix, labels = file2matrix('datingTestSet.txt')  # label 为字符串
    # print(labels)  # 'largeDose' 'smallDoses' 'didntLike'
    date_classify_test()
    pass
