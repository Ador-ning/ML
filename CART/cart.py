# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


# 分类与回归树
# 回归树--叶节点为单值 / 模型树-- 叶节点包含一个线性方程

class TreeNode(object):
    def __init__(self, feature, value, right, left):
        self.feature_split = feature
        self.value_split = value
        self.right_branch = right
        self.left_branch = left


def load_data(filename):
    data = []
    f = open(filename)
    for line in f.readlines():
        line_list = line.strip().split('\t')
        data.append([float(i) for i in line_list])
    return data


def bin_split_data(data, feature, value):
    """
    :param data:
    :param feature:
    :param value:
    :return:
    """
    # print(data[:, feature])  # 取行索引
    mat0 = data[np.nonzero(data[:, feature] > value)[0], :]
    mat1 = data[np.nonzero(data[:, feature] <= value)[0], :]
    return mat0, mat1


def reg_leaf(data):
    """
    # 无法分割时，求数据平均值，做叶子结点
    :param data:
    :return:
    """
    return np.mean(data[:-1])


def reg_err(data):
    """
    # 均方误差
    :param data:
    :return:
    """
    return np.var(data[:, -1] * np.shape(data)[0])


def choose_best_split(data, leaf_type, error_type, ops):
    """
    :param data:
    :param leaf_type:
    :param error_type:
    :param ops:
    :return:
    """
    tol_s = ops[0]  # 容许误差下降值
    tol_n = ops[1]  # 切分最少样本树
    # print(data[:-1].T)
    # print(data[:-1].T[0])

    if len(set(np.array(data[:, -1].T)[0])) == 1:  # 如果所有测试值相同
        return None, leaf_type(data)
    m, n = np.shape(data)
    s = error_type(data)
    best_s = np.inf
    best_feature = 0
    best_value = 0

    for feature in range(n - 1):  # 找出，best_feature and best_value, error after split
        for split_value in set(np.array(data[:, feature].T)[0]):
            mat0, mat1 = bin_split_data(data, feature, split_value)

            if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):
                continue
            new_s = error_type(mat0) + error_type(mat1)
            if new_s < best_s:
                best_feature = feature
                best_s = new_s
                best_value = split_value
    if (s - best_s) < tol_s:  # 如果误差减少不大，退出
        return None, leaf_type(data)
    mat0, mat1 = bin_split_data(data, best_feature, best_value)

    if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):  # 不满足，再分割条件
        return None, leaf_type(data)

    return best_feature, best_value


def create_tree(data, leaf_type=reg_leaf, error_type=reg_err, ops=(4, 10)):
    # 选择最佳分割
    feature, value = choose_best_split(data, leaf_type, error_type, ops)

    if feature is None:
        return value
    ret_tree = dict()
    # 保存分割相关信息
    ret_tree['split_feature'] = feature
    ret_tree['split_value'] = value
    left_data, right_data = bin_split_data(data, feature, value)
    ret_tree['left_data'] = create_tree(left_data, leaf_type, error_type, ops)
    ret_tree['right_data'] = create_tree(right_data, leaf_type, error_type, ops)
    return ret_tree


if __name__ == '__main__':
    data_ = load_data('ex0.txt')
    # print(data_)
    print(create_tree(np.matrix(data_))['split_value'])
    print(create_tree(np.matrix(data_))['left_data'])
    print(create_tree(np.matrix(data_))['right_data'])

    # pass
