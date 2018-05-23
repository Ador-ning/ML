# coding:utf-8

import numpy as np

"""
流程：
    去除平均值
    计算协方差举证 --> 特征值 和 特征向量
    将特征值从大到小排序
    保留最上面的 N 个特征向量
    将数据转换到上述 N 个特征向量构建的新空间中

第一个主成分就是 从方差最大的方向提取出来
第二个主成分来自 数据差异（方差）次大的方向，并且该方向与第一个主成分方向正交
"""


def load_data(filename, delim='\t'):
    fr = open(filename)
    data = []
    for line in fr.readlines():
        line_list = line.strip().split(delim)
        # string -> float
        data.append([float(elem) for elem in line_list])
    return np.matrix(data)


def pca(data_matrix, top_n_feature=9999):
    mean_values = np.mean(data_matrix, axis=0)  # 列
    mean_removed = data_matrix - mean_values  # 去平均值

    cov_matrix = np.cov(mean_removed, rowvar=0)  # 计算协方差矩阵
    eig_values, eig_vectors = np.linalg.eig(np.matrix(cov_matrix))  # 计算 特征值 和 特征向量
    eig_sort = np.argsort(eig_values)
    eig = eig_sort[:-(top_n_feature + 1):-1]  # 排新 选择 top N
    red_eig_vectors = eig_vectors[:, eig]  # 选择特征向量
    low_data = mean_removed * red_eig_vectors  # 将原数据进行转换
    recon_matrix = (low_data * red_eig_vectors.T) + mean_values
    return low_data, recon_matrix


if __name__ == '__main__':
    print(load_data('testSet.txt'))
    pass
