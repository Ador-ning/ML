# coding:utf-8

"""
1、节点类的实现 Node ：负责记录和维护节点自身信息以及这个节点相关的上下游连接，实现输出值和误差项的计算。如下：
    layer_index --- 节点所属的层的编号
    node_index --- 节点的编号
    downstream --- 下游节点 list
    upstream  ---- 上游节点 list
    output    ---- 节点的输出值
    delta   ------ 节点的误差项

2、ConstNode 类，偏置项类的实现：实现一个输出恒为 1 的节点（计算偏置项的时候会用到），如下：
    layer_index --- 节点所属层的编号
    node_index ---- 节点的编号
    downstream ---- 下游节点 list
        没有记录上游节点，因为一个偏置项的输出与上游节点的输出无关
    output    ----- 偏置项的输出

3、layer 类，负责初始化一层。作为的是 Node 节点的集合对象，提供对 Node 集合的操作。也就是说，layer 包含的是 Node 的集合。
    layer_index ---- 层的编号
    node_count ----- 层所包含的节点的个数
    def set_ouput() -- 设置层的输出，当层是输入层时会用到
    def calc_output -- 计算层的输出向量，调用的 Node 类的 计算输出 方法

4、Connection 类：负责记录连接的权重，以及这个连接所关联的上下游节点，如下：
    upstream_node --- 连接的上游节点
    downstream_node -- 连接的下游节点
    weight   -------- random.uniform(-0.1, 0.1) 初始化为一个很小的随机数
    gradient -------- 0.0 梯度，初始化为 0.0
    def calc_gradient() --- 计算梯度，使用的是下游节点的 delta 与上游节点的 output 相乘计算得到
    def get_gradient() ---- 获取当前的梯度
    def update_weight() --- 根据梯度下降算法更新权重

5、Connections 类：提供对 Connection 集合操作，如下：
    def add_connection() --- 添加一个 connection

6、Network 类：提供相应的 API，如下：
    connections --- Connections 对象
    layers -------- 神经网络的层
    layer_count --- 神经网络的层数
    node_count  --- 节点个数
    def train() --- 训练神经网络
    def train_one_sample() --- 用一个样本训练网络
    def calc_delta() --- 计算误差项
    def update_weight() --- 更新每个连接权重
    def calc_gradient() --- 计算每个连接的梯度
    def get_gradient() --- 获得网络在一个样本下，每个连接上的梯度
    def predict() --- 根据输入的样本预测输出值
"""

import random
from functools import reduce
from numpy import *


def sigmoid(x):
    """
    # sigmoid函数实现
    :param x: 输入向量
    :return:
    """
    return 1.0 / (1 + exp(-x))


# 神经元节点类
class Node(object):
    """
    # 定义神经网络的节点类
    """

    def __index__(self, layer_index, node_index):
        """
        # 初始化节点
        :param layer_index: 层的索引，表示层数
        :param node_index: 节点索引
        :return: None
        """
        self.layer_index = layer_index  # 设置层数
        self.node_index = node_index  # 设置层中的节点索引
        self.downstream = []  # 设置节点的下游节点， 下一层相连节点
        self.upstream = []  # 设置节点的上游节点， 上一层相连节点
        self.output = 0  # 该节点的输出
        self.delta = 0  # 该节点真实值与计算值之间的差值

    # 更改
    def set_output(self, output):
        """
        # 设置节点的 output
        :param output:
        :return:
        """
        self.output = output

    def append_downstream_connection(self, conn):
        """
        # 添加此节点的下游节点的连接
        :param conn: 当前节点的下游节点的list
        :return: None
        """
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """
        # 添加此节点的上游节点的连接
        :param conn: 当前节点的上游节点的list
        :return: None
        """
        self.upstream.append(conn)

    def calc_output(self):
        """
        # 计算节点输出
        :return:  output = sigmoid(W.T X)
        """
        # 使用 reduce() 函数对其中的因素求和
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0.0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        """
        # 计算隐藏层节点的 delta
        :return:
        """
        # 计算隐藏层的 delta
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream,
                                  0.0)
        # 计算此节点的 delta
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        """
        # 计算输出层的 delta
        :label: 输入向量对应的真实标签
        :return: None
        """
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        """
        # 将节点的信息打印
        :return: None
        """
        node_str = 'layer-node: %u-%u; output-%f, delta-%f' % \
                   (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream: ' + downstream_str + '\n\tupstream: ' + upstream_str


# 常节点类--偏置项
class ConstNode(object):
    """
    # 常数项节点对象，相当于计算的时候的偏置项
    """

    def __index__(self, layer_index, node_index):
        """
        # 初始化节点对象
        :param layer_index: 层的索引，表示层数
        :param node_index: 节点索引
        :return: None
        """
        self.layer_index = layer_index  # 设置层数
        self.node_index = node_index  # 设置层中的节点索引
        self.downstream = []  # 设置节点的下游节点， 下一层相连节点
        self.output = 1  # 该节点的输出

    def append_downstream_connection(self, conn):
        """
        # 添加此节点的下游节点的连接
        :param conn: 当前节点的下游节点的list
        :return: None
        """
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        """
        # 计算隐藏层节点的 delta
        :return:
        """
        # 计算隐藏层的 delta
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream,
                                  0.0)
        # 计算此节点的 delta
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        """
        # 将节点的信息打印
        :return: None
        """
        node_str = 'layer-node: %u-%u; output-%f, delta-%f' % \
                   (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream: ' + downstream_str


# 神经网络层对象，负责初始化一层，作为 Node类对象的集合， 提供 Node 集合的操作
class Layer(object):
    """
    # 神经网络的 Layer 类
    """

    def __init__(self, layer_index, node_count):
        """
        # Layer类对象初始化
        :param layer_index: 层索引
        :param node_count: 包含神经元个数
        """
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        """
        # 设置层的输出，当层是输入层是会用到
        :param data: 输出值的list
        :return: None
        """
        # 设置输入层中各个节点的 output
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        """
        # 计算层的输出向量
        :return: None
        """
        # 遍历本层所有节点（除偏置常节点）
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        """
        # 将层的信息打印
        :return:
        """
        # 依次层打印节点
        for node in self.nodes:
            print(node)  # __str__


# Connection 对象类，主要负责记录连接的权重，以及这个连接所关联的上下层节点
class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        """
        # 初始化 Connection对象
        :param upstream_node: 上层节点
        :param downstream_node: 下层节点
        """
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)  # 设置权重
        self.gradient = 0.0  # 设置梯度

    def calc_gradient(self):
        """
        # 计算梯度
        :return:
        """
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        """
        # 更新权重
        :param rate: 学习率
        :return:
        """
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        """
        # 当前梯度
        :return:
        """
        return self.gradient

    def __str__(self):
        """
        # 将连接信息打印
        :return:
        """
        return 'up: (%u-%u) -> down: (%u-%u) = weight: %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


# Connections对象类，提供 Connection 集合操作
class Connections(object):
    def __init__(self):
        """
        # Connections 类对象初始化
        """
        self.connections = []

    def add_connection(self, connection):
        """
        # 将 connection 添加到 Connections中
        :param connection:
        :return:
        """
        self.connections.append(connection)

    def dump(self):
        """
        # 将 Connections 的节点信息打印出来
        :return:
        """
        # 遍历
        for conn in self.connections:
            print(conn)


# Network对象，提供API
class Network(object):
    def __init__(self, layers):
        """
        # 初始化全连接神经网络
        :param layers: -- 二维数组，描述神经网络的每层节点
        """
        # 初始化 connections
        self.connections = Connections()
        self.layers = []
        layers_count = len(layers)  # 层数
        node_count = 0  # 节点数

        for i in range(layers_count):  # 遍历所有层，将每层信息添加到 layers中去
            self.layers.append(Layer(i, layers[i]))

        # 遍历出去输出层之外的所有层， 将连接信息添加到 connections对象中
        for layer in range(layers_count - 1):
            connections = [Connection(upstream_node, downstream_node) for upstream_node in self.layers[layer].nodes for
                           downstream_node in self.layers[layer + 1].nodes]
            # 遍历，添加 conn 到 connections
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data, learning_rate, epoch):
        """
        # 训练神经网络
        :param labels: 数据标签
        :param data: 数据
        :param learning_rate: 学习率
        :param epoch: 轮数
        :return: None
        """

        for i in range(epoch):  # 迭代 epoch 次
            for d in range(len(data)):  # 遍历每个样本
                self.train_sample(labels[d], data[d], learning_rate)  # 一个样本实例训练

    def train_sample(self, label, data, learning_rate):
        """
        # 内部函数， 训练单个样本
        :param label: 样本标签
        :param data: 样本
        :param learning_rate: 学习率
        :return: None
        """
        self.predict(data)
        self.calc_delta(label)
        self.update_weights(learning_rate)

    def predict(self, sample):
        """
        # 预测
        :param sample: 样本
        :return:
        """
        self.layers[0].set_output(sample)  # 为输入层的输出值，设置为样本向量，即不发生任何变化
        # 按层遍历
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[-1]))

    def calc_gradient(self):
        """
        # 计算每个连接的梯度
        :return: None
        """
        # 按照正常顺序遍历除输出层之外的所有层
        for layer in self.layers[:-1]:
            for node in layer.nodes:  # 遍历层中所有节点
                for conn in node.downstream:  # 遍历节点的下层连接
                    # 计算梯度
                    conn.cal_gradient()

    def get_gradient(self, label, sample):
        """
        # 获取网络在一个样本下，每个连接的梯度
        :param label: 样本标签
        :param sample: 样本
        :return: None
        """
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def calc_delta(self, label):
        """
        # 计算每个节点的 delta
        :param label: 样本真实值
        :return: None
        """
        output_nodes = self.layers[-1].nodes  # 获取输出层的所有节点

        # 遍历所有的 label
        for i in range(len(label)):  # 计算输出层节点的 delta
            output_nodes.calc_output_layer_delta(label[i])

        for layer in self.layers[-2::-1]:  # [-2::-1]就是将 layers 倒过来
            for node in layer.nodes:  # 遍历每层节点
                node.calc_hidden_layer_delta()  # 计算隐藏层的 delta

    def update_weights(self, learning_rate):
        """
        # 更新权重
        :param learning_rate: 学习率
        :return: None
        """
        # 遍历
        for layer in self.layers[:-1]:
            for node in layer.nodes:  # 遍历每层的所有节点
                for conn in node.downstream:  # 遍历节点的下层节点
                    conn.update_weight(learning_rate)  # 更新

    def dump(self):
        """
        # 打印信息
        :return:
        """
        for layer in self.layers:
            layer.dump()


class Normalizer(object):
    """
    # 归一化
    """

    def __init__(self):
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]  # 判断位

    def norm(self, number):
        """
        # 对 number 进行规范化
        :param number:
        :return:
        """
        # 0.9 or 0.1
        return list(map(lambda m: 0.9 if number & m else 0.1, self.mask))

    def de_norm(self, vec):
        """
        # 对向量反规范化
        :param vec:
        :return:
        """
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))  # 二分类
        for m in range(len(self.mask)):  # 遍历
            binary[m] = binary[m] * self.mask[m]
        # 将结果相加得到最终的预测结果
        return reduce(lambda x, y: x + y, binary)


def mean_square_error(vec1, vec2):
    """
    # MSE
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 1/2 * (x-y)^2
    """
    return 0.5 * reduce(lambda a, b: a + b, map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2)))


def train_data_set():
    # 调用 Normalizer() 类
    normalizer = Normalizer()
    # 初始化一个 list，用来存储后面的数据
    data_set = []
    labels = []
    # 0 到 256 ，其中以 8 为步长
    for i in range(0, 256, 8):
        # 调用 normalizer 对象的 norm 方法
        n = normalizer.norm(int(random.uniform(0, 256)))
        # 在 data_set 中 append n
        data_set.append(n)
        # 在 labels 中 append n
        labels.append(n)
    # 将它们返回
    return labels, data_set


if __name__ == '__main__':
    pass
