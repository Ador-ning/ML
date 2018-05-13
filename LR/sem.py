# coding:utf-8
"""
痛过垃圾短信数据训练 Logistic Regression model, 并进行留存交叉验证
"""

import re
import random
import numpy as np
import matplotlib.pyplot as plt
from logreg_sga import lr_classifier_s

ENCODING = 'ISO-8859-1'
TRAIN_PERCENTAGE = 0.9


def get_doc_vector(words, vocabulary):
    """
    # 根据词汇表将文档中的词条转换成文档向量
    :param words: 文档中的词条列表
    :param vocabulary: 总的词汇表
    :return doc_vector: 用于贝叶斯分类的文档向量
    """
    doc_vector = [0] * len(vocabulary)

    for word in words:
        if word in vocabulary:
            idx = vocabulary.index(word)
            doc_vector[idx] += 1

    return doc_vector


def parse_line(line):
    """
    # 解析数据集中的每一条返回词向量和短信类型
    :param line:
    :return:
    """
    class1 = line.split(',')[-1].strip()  # 取 label
    content = ','.join(line.split(',')[:-1])  # 取短信内容
    word_vector = [word.lower() for word in re.split(r'\W+', content) if word]  # 提取短信单词，去掉无关符号， 将大写转小写
    return word_vector, class1


def parse_file(filename):
    """
    # 解析文件中的数据
    :param filename:
    :return:
    """
    vocabulary, word_vectors, classes = [], [], []

    with open(filename, 'r', encoding=ENCODING) as f:
        for line in f:
            if line:
                word_vector, class1 = parse_line(line)
                vocabulary.extend(word_vector)
                word_vectors.append(word_vector)
                classes.append(class1)
    vocabulary = list(set(vocabulary))
    return vocabulary, word_vectors, classes


if __name__ == '__main__':
    vocabulary, word_vectors, classes = parse_file('english_big.txt')  # 数据
    clf = lr_classifier_s()  # 模型

    # 设置训练集 和 测试集
    n_test = int(len(classes) * (1 - TRAIN_PERCENTAGE))

    test_word_vectors = []
    test_classes = []
    for i in range(n_test):
        idx = random.randint(0, len(word_vectors) - 1)
        test_word_vectors.append(word_vectors.pop(idx))
        test_classes.append(classes.pop(idx))

    train_word_vectors = word_vectors
    train_class = classes

    # 将类型标签改为 0/1
    l = lambda x: 1 if x == 'spam' else 0
    train_class = list(map(l, train_class))
    test_classes = list(map(l, test_classes))

    # 结果
    train_data = [get_doc_vector(word, vocabulary) for word in train_word_vectors]

    # 训练模型
    clf.stochastic_gradient_ascent(train_data, train_class)

    # 测试模型
    error = 0
    for test_word_vector, test_cls in zip(test_word_vectors, test_classes):
        test_data = get_doc_vector(test_word_vector, vocabulary)
        predict_cls = clf.classify(test_data)
        if predict_cls != test_cls:
            print('Predict: {} -- Actual: {}'.format(predict_cls, test_cls))
            error += 1

    print("Error rate: {}".format(error / len(test_classes)))
