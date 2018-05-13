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
    :return doc_vector:
    """
