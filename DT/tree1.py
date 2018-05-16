# coding:utf-8

from math import log
import operator


def create_data():
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'],
                [0, 1, 'no'], [0, 1, 'no']]
    feature_names = ['no surfacing', 'flippers']
    return data_set, feature_names


# 计算一个特征值
def count_entropy(data_set):
    """
    # 计算信息熵，数据集中，最后一位数据， 需要划分数据 / 或者计算 base entropy
    :param data_set:
    :return:
    """
    number_entries = len(data_set)
    # print(number_entries)
    label_counts = {}
    for feature_vector in data_set:
        # print(feature_vector)
        current_label = feature_vector[-1]  # #### 做键值
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key] / number_entries)
        entropy -= prob * log(prob, 2)
    return entropy


def split_data(data_set, axis, value):
    """
    # 根据feature 及其 feature取值 划分数据集
    :param data_set: 数据
    :param axis: feature 在样本中的下标
    :param value: feature 取值
    :return:
    """
    ret_data = []
    for feature_vector in data_set:
        if feature_vector[axis] == value:
            reduce_feature = feature_vector[:axis]
            # list会出现越界下标越界
            reduce_feature.extend(feature_vector[axis + 1:])  # extend
            ret_data.append(reduce_feature)
    return ret_data


def choose_best_feature_split(data_set):
    """
    # 选择数据集 信息增益最大的 feature
    :param data_set:
    :return:
    """
    number_feature = len(data_set[0]) - 1  # 含分类结果标签

    base_entropy = count_entropy(data_set)  # base entropy
    print(base_entropy)
    best_info_gain = 0.0  # 信息增益
    best_feature = -1

    # 依次计算 feature 熵增益
    for i in range(number_feature):
        feature_list = [example[i] for example in data_set]  # feature i, 取值

        unique_values = set(feature_list)  # 建立集合
        new_entropy = 0.0

        for value in unique_values:
            sub_data = split_data(data_set, i, value)  # 根据feature取值 划分feature i
            prob = len(sub_data) / float(len(data_set))
            new_entropy += prob * count_entropy(sub_data)

        info_gain = base_entropy - new_entropy  # 增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(feature_list):
    """
    # 返回出现次数最多的分类名称
    :param feature_list: 分类名称列表
    :return:
    """
    feature_count = {}
    for vote in feature_list:
        if vote not in feature_count.keys():
            feature_count[vote] = 0
    sort_feature_count = sorted(feature_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    return sort_feature_count[0][0]


# #### 过程
def create_tree(data_set, feat):
    class_list = [example[-1] for example in data_set]  # 样本分类结果

    # 只有一类
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 只有一个 类别
    if len(data_set[0]) == 1:
        return majority_count(data_set)

    best_feature = choose_best_feature_split(data_set)
    best_feature_label = feat[best_feature]

    my_tree = {best_feature_label: {}}
    del (feat[best_feature])
    feat_values = [example[best_feature] for example in data_set]
    unique_values = set(feat_values)

    for value in unique_values:
        sub_feat = feat[:]
        my_tree[best_feature_label][value] = create_tree(split_data(data_set, best_feature, value), sub_feat)

    return my_tree


def classify(input_tree, feature_label, test_vector):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feature_index = feature_label.index(first_str)
    class_label = ' '
    for key in second_dict.keys():
        if test_vector[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_label, test_vector)
            else:
                class_label = second_dict[key]

    return class_label


if __name__ == '__main__':
    data, labels = create_data()
    # count_entropy(data)
    print(create_tree(data, labels))
    pass
