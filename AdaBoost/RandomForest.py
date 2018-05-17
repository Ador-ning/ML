# coding:utf-8
import numpy as np


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


if __name__ == '__main__':
    load_data('sonar-all-data.txt')
    pass
