#!/usr/bin/env python
__coding__ = 'utf-8'

import numpy as np
import random
from time import sleep
from matplotlib import pyplot as plt


def load_data(filename):
    data = []
    f = open(filename)
    for line in f.readlines():
        line_list = line.strip().split('\t')
        data.append([float(i) for i in line_list])
    return data


def distance_count(vec_a, vec_b):
    """
    # 计算向量间的 欧氏距离
    :param vec_a: np.array
    :param vec_b: np.array
    :return:
    """
    return np.sqrt(sum(np.power(vec_a - vec_b, 2)))


def rand_center(data, k):
    n = np.shape(data)[1]  # feature numbers
    center = np.matrix(np.zeros((k, n)))  # (k, n)

    for i in range(n):
        min_i = data[:, i].min()
        range_i = float(data[:, i].max() - min_i)  # feature value range
        for j in range(k):
            center[j, i] = min_i + range_i * (random.randint(1, k) / k)
    return center


def k_means(data, k, dis_method=distance_count, create_center=rand_center):
    data = np.matrix(data)
    m = np.shape(data)[0]
    center = create_center(data, k)  # 随机初始化

    cluster = np.matrix(np.zeros((m, 2)))
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_distance = np.inf
            min_index = -1
            # 找最近的质心，并记录下来
            for j in range(k):
                dist = dis_method(np.array(center[j, :])[0], np.array(data[i, :])[0])
                if dist < min_distance:
                    min_distance = dist
                    min_index = j

            if cluster[i, 0] != min_index:
                cluster_changed = True  # 如果该实例的簇分配发生变化
            cluster[i, :] = min_index, min_distance ** 2  # 存储 实例与质心 的距离信息
            print(center)

        for cent in range(k):  # 更新质心位置
            pts_in_cluster = data[np.nonzero(cluster[:, 0].A == cent)[0]]  # 过滤
            center[cent, :] = np.mean(pts_in_cluster, axis=0)  # 列均值
    # #### 异常 nan 值
    return center, cluster


def bik_k_means(data, k, dis_method=distance_count):
    data = np.matrix(data)
    m = np.shape(data)[0]

    cluster = np.matrix(np.zeros((m, 2)))
    center = np.mean(data, axis=0).tolist()  # begin cluster vector
    center_list = [center]

    for j in range(m):
        cluster[j, 1] = dis_method(np.array(center)[0], np.array(data[j, :])[0]) ** 2

    while len(center_list) < k:
        lowest_sse = np.inf
        for i in range(len(center_list)):
            pts_in_curr_cluster = data[np.nonzero(cluster[:, 0].A == i)[0], :]  # 功能
            center_matrix, split_cluster = k_means(pts_in_curr_cluster, 2)
            sse_split = sum(split_cluster[:, 1])
            sse_not_split = sum(cluster[np.nonzero(cluster[:, 0].A != i)[0], 1])

            if sse_not_split + sse_split < lowest_sse:
                best_cent_split = i
                best_new_cent = center_matrix
                best_cluster_ass = split_cluster.copy()
                lowest_sse = sse_split + sse_not_split
        # 找出最好的簇分配结果
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0].A == 1)[0], 0] = len(center_list)
        # 更新为最佳质心
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0].A == 0)[0], 0] = best_cent_split

        # 更新质心列表
        # 更新原质心中的第i个质心，为使用二分k_means后 best_new_cent的第一个质心
        center_list[best_cent_split] = best_new_cent[0, :].tolist()
        # 添加 best_new_cent的第二个质心
        center_list.append(best_new_cent[1, :].tolist())
        # 重新分配最好簇下的数据(质心)以及sse
        cluster[np.nonzero(cluster[:, 0].A == best_cent_split)[0], :] = best_cluster_ass
    return np.matrix(center_list), cluster


if __name__ == '__main__':
    d = load_data('testSet.txt')
    # print(rand_center(np.matrix(d), 5))
    # k_means(d, 4)
    bik_k_means(d, 3)
