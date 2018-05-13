# coding:utf-8

import copy
import uuid
import pickle
from collections import defaultdict, namedtuple
from math import log2


class DecisionTree(object):
    """
    使用 ID3 算法划分数据集 DT 分类器
    """

    @staticmethod
    def split_data(data, classes, feature_idx):
        """
        # 根据某个特征以及特征值划分数据集
        :param data: 待划分的数据集， 由数据向量组成的列表
        :param classes: 数据集对应的类型，同数据集长度
        :param feature_idx: 特征在特征向量中的索引
        :return: 保存分割后数据的字典 特征值：[子数据集，子类型列表]
        """
        split_dict = {}
        for data_vector, class1 in zip(data, classes):
            feature_val = data_vector[feature_idx]
            sub_data, sub_class = split_dict.setdefault(feature_val, [[], []])
            sub_data.append(data_vector[:feature_idx] + data_vector[feature_idx + 1:])
            sub_class.append(class1)
        return split_dict

    @staticmethod
    def get_entropy(values):
        """
        # 计算信息熵
        :param values:
        :return:
        """
        uniq_val = set(values)
        val_nums = {key: values.count(key) for key in uniq_val}
        probs = [v / len(values) for k, v in val_nums.items()]
        entropy = sum([-prob.log2(prob) for prob in probs])
        return entropy

    def choose_best_split_featureure(self, data_set, classes):
        """
        # 根据 信息增益 确定最好的划分数据的特征
        :param data_set: 待划分的数据
        :param classes: 数据集对应的类型
        :return: 划分数据的 信息增益最大 特征索引
        """
        base_entropy = self.get_entropy(classes)

        feature_num = len(data_set[0])
        entropy_gains = []

        for i in range(feature_num):
            split_dict = self.split_data(data_set, classes, i)

            new_entropy = sum(
                [len(subclass) / len(classes) * self.get_entropy(subclass) for _, (_, subclass) in split_dict.items()])
            entropy_gains.append(base_entropy - new_entropy)
        return entropy_gains.index(max(entropy_gains))

    @staticmethod
    def get_majority(classes):
        """
        # 返回类型中占大多数的类型
        :param classes:
        :return:
        """
        cls_num = defaultdict(lambda: 0)
        for cls in classes:
            cls_num[cls] += 1
        return max(cls_num, key=cls_num.get())

    def create_tree(self, data_set, classes, feature_names):
        """
        # 跟据当前数据集递归创建 DT
        :data_set: 数据集
        :classes: 数据集中数据相应的类型 result label
        :feature_names: 数据集中数据对应的特征名称 feature names
        :return: 以字典形式返回 DT
        """
        # 如果数据集中只有一种 类型, 停止
        if len(set(classes)) == 1:
            return classes[0]

        # 如果遍历完所有特征，返回比例最多的类型
        if len(feature_names) == 0:
            return self.get_majority(classes)

        # 分裂创建新的子树
        tree = {}
        best_feature_idx = self.choose_best_split_featureure(data_set, classes)
        feature = feature_names[best_feature_idx]
        tree[feature] = {}

        # 创建用于递归创建子树的子数据集
        sub_feature_names = feature_names[:]
        sub_feature_names.pop(best_feature_idx)  # feature

        split_dict = self.split_data(data_set, classes, best_feature_idx)  # data

        for feature_val, (sub_data_set, sub_classes) in split_dict.items():
            tree[feature][feature_val] = self.create_tree(sub_data_set, sub_classes, sub_feature_names)

        self.tree = tree
        self.feature_names = feature_names
        return tree

    def get_nodes_edges(self, tree=None, root_node=None):
        """
        # 返回树中所有节点和边
        :param tree:
        :param root_node:
        :return:
        """
        Node = namedtuple('Node', ['id', 'label'])
        Edge = namedtuple('Edge', ['start', 'end', 'label'])

        if tree is None:
            tree = self.tree

        if type(tree) is not dict:
            return [], []

        nodes, edges = [], []

        if root_node is None:
            label = list(tree.keys())[0]
            root_node = Node._make([uuid.uuid4(), label])
            nodes.append(root_node)

        for edge_label, sub_tree in tree[root_node.label].item():
            node_label = list(sub_tree.keys())[0] if type(sub_tree) is dict else sub_tree
            sub_node = Node._make([uuid.uuid4(), node_label])
            nodes.append(sub_node)

            edge = Edge._make([root_node, sub_node, edge_label])
            edges.append(edge)

            # 递归
            sub_nodes, sub_edges = self.get_nodes_edges(sub_tree, root_node=sub_node)
            nodes.append(sub_nodes)
            edges.append(sub_edges)

        return nodes, edges

    def dotify(self, tree=None):
        """
        # 获取树的 Graphviz Dot文件的内容
        :param tree:
        :return:
        """
        pass

    def classify(self, data_vector, feature_names=None, tree=None):
        """
        # 根据构建的决策树对数据进行分类
        :param data_vector:
        :param feature_names:
        :param tree:
        :return:
        """
        if tree is None:
            tree = self.tree

        if feature_names is None:
            feature_names = self.feature_names

        #
        if type(tree) is not dict:
            return tree

        feature = list(tree.keys())[0]
        value = data_vector[feature_names.index(feature)]
        sub_tree = tree[feature][value]
        # 递归
        return self.classify(data_vector, feature_names, sub_tree)

    def dump_tree(self, file_name, tree=None):
        """
        # 存储决策树
        :param file_name:
        :param tree:
        :return:
        """
        if tree is None:
            tree = self.tree

        with open(file_name, 'wb') as f:
            pickle.dump(tree, f)

    def load_tree(self, file_name):
        """
        # load tree model
        :param file_name:
        :return:
        """
        with open(file_name) as f:
            tree = pickle.load(f)
            self.tree = tree
        return tree
