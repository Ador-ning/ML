# coding:utf-8
import numpy as np
import re
import random


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage '],
                    ['mr', 'licks', 'ate ', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]  # 1 侮辱
    return posting_list, class_vector


# 将单词去重放入 list
def create_vocabulary_list(data_set):
    vocabulary_set = set([])
    for document in data_set:
        vocabulary_set = vocabulary_set | set(document)  # 并集
    return list(vocabulary_set)


# 将词组，转化为向量 --> 统计频率
def set_word2vec(vocabulary_list, input_set):
    ret_vector = [0] * len(vocabulary_list)

    for word in input_set:
        if word in vocabulary_list:
            ret_vector[vocabulary_list.index(word)] = 1
        else:
            print('The word: %s not in vocabulary!' % word)

    return ret_vector


def train_nb(train_matrix, train_category):
    """
    :param train_matrix: 训练文档矩阵
    :param train_category: 文档对应的类别
    :return:
    """
    num_train_docs = len(train_matrix)  # 训练文档数
    number_words = len(train_matrix[0])  # 词库单词数
    pAbusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(number_words)  # array 类型
    p1_num = np.ones(number_words)
    p0_Denom = 2.0
    p1_Denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 1:  # 侮辱性文档
            p1_num += train_matrix[i]
            p1_Denom += sum(train_matrix[i])  # 频率
        else:
            p0_num += train_matrix[i]
            p0_Denom += sum(train_matrix[i])
    p1_vector = np.log(p1_num / p1_Denom)
    p0_vector = np.log(p0_num / p0_Denom)
    return p0_vector, p1_vector, pAbusive


# 实现概率公式  #### 简单实现
def classify_nb(vec2classify, p0_vector, p1_vector, p_class):
    p1 = sum(vec2classify * p1_vector) + np.log(p_class)  # 1类概率
    p0 = sum(vec2classify * p0_vector) + np.log(p_class)

    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    list_post, list_class = load_data_set()
    vocabulary = create_vocabulary_list(list_post)
    train_matrix = []
    for i_doc in list_post:
        train_matrix.append(set_word2vec(vocabulary, i_doc))

    p0_V, p1_V, pAb = train_nb(train_matrix, list_class)

    test_entry = ['love', 'stupid', 'dog']
    test_vector = np.array(set_word2vec(vocabulary, test_entry))
    print(test_entry, "classified as:", classify_nb(test_vector, p0_V, p1_V, pAb))


# naive bayes 词袋子模型
def bag_of_words2vec(vocabulary_list, input_set):
    return_vector = [0] * len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index(word)] += 1
    return return_vector


# 文本解析
def text_parse(string):
    list_of_tokens = re.split(r'\w*', string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


# ### text 文本编码 --> 读取数据有问题 open()
def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)  # 词库
        full_text.extend(word_list)  # 文档库
        class_list.append(1)
        print(i)
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocabulary_list = create_vocabulary_list(doc_list)
    training_set = range(50)
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])

    train_matrix = []
    train_class = []
    for doc_index in training_set:
        train_matrix.append(set_word2vec(vocabulary_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])

    p0_v, p1_v, p_spam = train_nb(np.array(train_matrix), np.array(train_class))
    error_count = 0
    for doc_index in test_set:
        word_vector = set_word2vec(vocabulary_list, doc_list[doc_index])
        if classify_nb(np.array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print('The error rate is %f' % (float(error_count) / len(test_set)))


if __name__ == '__main__':
    # testing_nb()
    spam_test()
