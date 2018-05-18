# coding:utf-8
import numpy as np
from random import seed, random, randrange


# Random Forest
# 模型融合： bagging方法： 并行/抽样数据


def load_data(filename):
    """
    :param filename:
    :return: list type
    """
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line:
                continue
            line_list = line.strip().split(',')  # all data is str type, convert into float at matrix
            """
            for col in line.split(','):
                str_ = col.strip()  # 移除字符串头尾指定的字符，生成新的字符串
                if str_.isdigit():
                    line_list.append(float(str_))
                else:
                    line_list.append(str_)
            """  # slow
            # print(type(line_list[0]))
            data.append(line_list)
    return data


def subsample(data, ratio):
    """
    # 创建数据集的随机子样本
    :param data: 数据
    :param ratio: 样本比例
    :return: 对样本进行随机抽样
    """
    sample = list()
    n_sample = round(len(data) * ratio)  # 作用？？？
    while len(sample) < n_sample:
        index = randrange(len(data))
        sample.append(data[index])

    return sample


def predict(node, row):
    """
    # 预测模型分类结果
    :param node:
    :param row:
    :return:
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def bagging_predicting(trees, row):
    """
    :param trees: 决策树集合
    :param row: 测试数据行
    :return: 返回随机森林中，决策树结果出现次数最大的
    """
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def test_split(index, value, sample):
    """
    # 根据特征和特征值 分割数据集
    :param index: 特征索引
    :param value: 特征值
    :param sample: 数据集
    :return: (left, right)
    """
    left, right = [], []
    for row in sample:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


def to_terminal(group):
    """
    # 输出group中出现次数比较多的标签
    :param group:
    :return:
    """
    out_comes = [row[-1] for row in group]  # 整理处 分类
    return max(set(out_comes), key=out_comes.count)


def get_split(sample, num_f):
    """
    # 计算，找出分裂数据的最有特征 index，特征值 row[index]， 以及分裂后的数据 group(left, right)
    :param sample: 数据
    :param num_f: 使用特征数目
    :return: 分裂的相关信息，
    """
    class_values = list(set(row[-1] for row in sample))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()

    while len(features) < num_f:  # 挑选特征 --> 随机
        index = randrange(len(sample[0]) - 1)
        if index not in features:
            features.append(index)

    for index in features:  # 在 num_f特征中，挑选最优特征
        for row in sample:
            # 遍历每行 index 索引下的特征值，作为分类值 value, 找出最优的分类特征和特征值
            groups = test_split(index, row[index], sample)
            # 注 。。。。。。。
            # 数据，划分，计算基尼指数？？？
            gini = gini_index(groups, class_values)

            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def gini_index(groups, class_values):
    """
    # 计算代价，分类越准确，gini值越小
    :param groups:
    :param class_values:
    :return:
    """
    gini = 0
    for class_value in class_values:  # [0, 1]
        for group in groups:  # (left, right)
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / size
            gini += (proportion * (1.0 - proportion))  # 求和 (1-P)*P
    return gini


def split(node, max_d, min_s, num_f, depth):
    """
    # 创建子分割器，递归分类，直到分类结束
    :param node: 待分割节点, dict
    :param max_d: 树最大深度
    :param min_s:
    :param num_f: 特征数量
    :param depth:
    :return:
    """
    left, right = node['groups']  # 分割节点两侧
    del (node['groups'])

    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # process left child
    if len(left) <= min_s:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, num_f)
        split(node['left'], max_d, min_s, num_f, depth + 1)

    # process right child
    if len(right) <= min_s:
        node['right'] = to_terminal(left)
    else:
        node['right'] = get_split(left, num_f)
        split(node['right'], max_d, min_s, num_f, depth + 1)


def build_tree(sample, max_d, min_s, num_f):
    """
    :param sample: 随机样本
    :param max_d: 树最大深度
    :param min_s: 叶子节点的大小
    :param num_f: 选取特征的个数
    :return: root 决策树
    """
    root = get_split(sample, num_f)  # 返回 最有列 --> 分裂点，和相关信息

    # 对左右两侧的数据进行递归调用，去掉已使用的特征
    split(root, max_d, min_s, num_f, 1)  # 1 -- 深度
    return root


def random_forest(data_train, data_test, max_d, min_s, sample_s, num_t, num_f):
    """
    # 评估算法，返回模型得分
    :param data_train: 训练数据
    :param data_test: 测试数据
    :param max_d: 决策树最大深度
    :param min_s: 叶子节点的大小
    :param sample_s: 训练数据集的样本比例
    :param num_t: 决策树的数量
    :param num_f: 使用特征的个数
    :return:
    """
    trees = list()

    for i in range(num_t):
        sample = subsample(data_train, sample_s)
        tree = build_tree(sample, max_d, min_s, num_f)
        trees.append(tree)

    predictions = [bagging_predicting(trees, row) for row in data_test]
    return predictions


def accuracy_metric(actual, predicted):
    """
    # 计算准确度
    :param actual: 实际结果
    :param predicted: 预测结果
    :return:
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100


def cross_validation_split(data, num_folds):
    """
    :param data:
    :param num_folds:
    :return: list集合
    """
    data_split = []
    data_copy = list(data)

    fold_size = len(data) / num_folds

    for i in range(num_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(data_copy))
            fold.append(data_copy[index])
        data_split.append(fold)
    return data_split


def evaluate_algorithm(data, algorithm, num_folds, *args):
    """
    # 评估模型
    :param data: 数据集
    :param algorithm: 算法
    :param num_folds: 数据份数
    :param args: 其他参数
    :return: 模型得分
    """
    folds = cross_validation_split(data, num_folds)  # 划分数据
    scores = list()

    # 每次取 一个fold作为测试集，其余训练
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train_set, test_set, *args)
        # print(predicted)
        actual = [row[-1] for row in fold]
        # print(actual)
        # 计算正确率
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


if __name__ == '__main__':
    data_set = load_data('sonar-all-data.txt')
    # print(np.matrix(data_set))

    n_folds = 5  # 5份数据进行交叉验证
    max_depth = 15  # 决策树深度
    min_size = 1  # 决策树的叶子节点最少的元素量
    sample_size = 1.0  # 做决策树时候的样本的比例

    # print(len(data_set[0]) - 1)  # 样本feature数目 # 60
    n_features = 30

    for num_trees in [1, 2, 3]:  # 随机森林树的数量
        score = evaluate_algorithm(data_set, random_forest, n_folds,
                                   max_depth, min_size, sample_size, num_trees, n_features)
        seed(1)
        # print('random=', random())
        print('trees: ', num_trees)
        print('scores:', score)
        print("mean accuracy: %.3f" % (sum(score) / float(len(score))))
