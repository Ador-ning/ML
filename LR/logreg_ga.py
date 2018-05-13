# coding:utf-8
import os
import numpy as np
import matplotlib.pyplot as plt


class lr_classifier(object):
    """
    使用梯度上升算法 LR 分类器
    """

    @staticmethod
    def sigmoid(x):
        """
        :param x:  sigmoid function
        :return:
        """
        return 1.0 / (1 + np.exp(-x))

    def gradient_ascent(self, data, labels, max_iter=1000):
        """
        :param data: 数据特征矩阵 M, N numpy matrix
        :param labels: 数据集对应的类型向量 N,1
        :param max_iter:
        :return:
        """
        data = np.matrix(data)  # (100, 3)
        label = np.matrix(labels).reshape(-1, 1)  # (100, 1)
        m, n = data.shape
        w = np.ones((n, 1))  # [1, 1, 1]   (100, 3) x (3, 1) -> (100, 1)

        alpha = 0.001
        ws = []
        for i in range(max_iter):
            error = label - self.sigmoid(data * w)  # (100, 1)  所有数据更新
            w += alpha * data.T * error  # (3, 100) x (100, 1) -> W (3, 1)
            ws.append(w.reshape(-1, 1).tolist()[0])
        self.w = w  # (3,1)
        return w, np.array(ws)

    def classify(self, data, w=None):
        """
        测试
        :param data:
        :param w:
        :return:
        """
        if w is None:
            w = self.w
        data = np.matrix(data)
        prob = self.sigmoid((data * w).tolist()[0][0])
        return round(prob)


def load_data(filename):
    data_set, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            split_line = [float(i) for i in line.strip().split('\t')]
            data, label = [1.0] + split_line[:-1], split_line[-1]
            # print([1.0] + split_line[:-1]) 第一列：全为1
            data_set.append(data)
            labels.append(label)

    data_set = np.matrix(data_set)
    labels = np.matrix(labels)
    return data_set, labels


# #### 有问题
def snap_shot(w, data_set, labels, picture_name):
    """
    绘制类型分割线图
    """
    if not os.path.exists('./snap_shots'):
        os.mkdir('./snap_shots')
    fig = plt.figure()
    ax = fig.add_subplot()

    pts = {}
    # for data, label in zip(data_set.tolist(), labels.tolist()):
    #     pts.setdefault(label, []).append(data)

    for j in range(len(labels)):
        pts.setdefault(labels.tolist()[0][j], []).append(data_set[j].tolist())

    for label, data in pts.items():
        # data = np.array(data)
        plt.scatter(data[0][0][1], data[0][0][2], label=label, alpha=0.5)

    # 分割线绘制
    def get_y(x, w):
        w0, w1, w2 = w
        return (-w0 - w1 * x) / w2

    x = [-4.0, 3.0]
    print(w)
    y = [get_y(i, w) for i in x]
    plt.plot(x, y, linewidth=2)

    picture_name = './snap_shots/{}'.format(picture_name)
    fig.savefig(picture_name)
    plt.close(fig)


if __name__ == '__main__':
    data_set, labels = load_data('testSet.txt')
    # print(data_set.shape, labels.shape) 100x3 1x100

    clf = lr_classifier()
    w, ws = clf.gradient_ascent(data_set, labels, max_iter=5000)  # (3, 1) (5000, 1)
    m, n = ws.shape

    '''
    for i in range(300):
        if i % 30 == 0:
            print('{}.png saved'.format(i))
            snap_shot(ws[i].tolist(), data_set, labels, '{}.png'.format(i))

    fig = plt.figure()
    for i in range(n):
        label = 'w{}'.format(i)
        ax = fig.add_subplot(n, 1, i + 1)
        ax.plot(ws[:, i], label=label)
        ax.legend()
    fig.savefig('w_tra.png')
    '''
