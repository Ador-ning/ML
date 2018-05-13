# coding:utf-8
from logreg_ga import lr_classifier, load_data
import numpy as np
import random


# 随机梯度上升
class lr_classifier_s(lr_classifier):

    def stochastic_gradient_ascent(self, data_set, labels, max_iter=5000):
        data_set = np.matrix(data_set)
        m, n = data_set.shape  # (m, n)
        w = np.matrix(np.ones((n, 1)))  # (n, 1)
        ws = []

        for i in range(max_iter):
            data_indices = list(range(m))
            random.shuffle(data_indices)  # 部分数据
            for j, id in enumerate(data_indices):
                data, label = data_set[id], labels[id]
                error = label - self.sigmoid((data * w).tolist()[0][0])
                alpha = 4 / (1 + j + i) + 0.01
                w += alpha * data.T * error
                ws.append(w.T.tolist()[0])

        self.w = w
        return w, np.array(ws)
