# coding:utf-8
import numpy as np
import random


def load_data(filename):
    data = []
    label = []
    f = open(filename)
    for line in f.readlines():
        line_list = line.strip().split('\t')
        data.append([float(line_list[0]), float(line_list[1])])
        label.append(float(line_list[2]))
    return data, label


# 有问题
def select_rand(i, m):
    """
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数目
    :return:
    """
    j = i
    while (j == i):  # 有问题
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    """
    # 调整 >h 或 >l 的alpha值
    :param aj:
    :param h:
    :param l:
    :return:
    """
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


def smo_simple(data_in, label_in, c, toler, max_iter):
    """
    :param data_in: 数据
    :param label_in: 标签
    :param c: 常数
    :param toler: 容错率
    :param max_iter: 最大循环次数
    :return:
    """
    data_matrix = np.matrix(data_in)  # (m, n)
    label_matrix = np.matrix(label_in).T  # (n, 1)
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.matrix(np.zeros((m, 1)))
    iter1 = 0

    while iter1 < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            # 预测类别
            fxi = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + b
            # 误差
            ei = fxi - float(label_matrix[i])

            # 如果alpha可以更改，进入优化程序
            # 测试正负间隔， 检查alpha -> 保证不能 == 0/c
            # alpha 小于0或大于C时， 后面调整至 0/c -- 边界
            if ((label_matrix[i] * ei < -toler) and (alphas[i] < c)) or \
                    ((label_matrix[i] * ei > toler) and (alphas[i] > c)):
                j = select_rand(i, m)  # 随机选择第二个alpha
                fxj = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :])) + b
                ej = fxj - float(label_matrix[j])

                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # 保证alpha在 0 和 C 之间
                if label_matrix[i] != label_matrix[j]:
                    ll = max(0, alphas[j] - alphas[i])
                    hh = min(c, c + alphas[j] - alphas[i])
                else:
                    ll = max(0, alphas[j] + alphas[i] - c)
                    hh = min(c, alphas[j] + alphas[i])

                if ll == hh:
                    print('ll == hh')
                    continue  # 对 alpha[j] 不调整
                # 是alpha[j]的最优修改量
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T \
                      - data_matrix[i, :] * data_matrix[i, :].T \
                      - data_matrix[j, :] * data_matrix[j, :]
                if eta >= 0:
                    print("eta > =0")
                    continue
                # 调整
                alphas[j] -= label_matrix[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], hh, ll)

                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not moving enough.")
                    continue

                # 对 i 进行修改，修改量与 J 相同，但方向相反
                alphas[i] += label_matrix[j] * label_matrix[i] * (alpha_j_old - alphas[j])
                # 设置常数项
                b1 = b - ei - label_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_matrix[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                b2 = b - ej - label_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_matrix[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T

                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter: %d, i: %d, pairs changed %d" % (iter1, i, alpha_pairs_changed))

        if alpha_pairs_changed == 0:
            iter1 += 1
        else:
            iter1 = 0

        print('iteration number: %d.' % iter1)

    return b, alphas


if __name__ == '__main__':
    data, label = load_data('testSet.txt')
    print(data)
