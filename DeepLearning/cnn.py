# coding:utf-8

import numpy as np
from activator import ReLUActivator, IdentityActivator


# 获取卷积区域
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    """
    # 从输入数组中获取本次卷积的区域
    # 自动适配输入为 2D / 3D 情况

    """
    start_i = i * stride
    start_j = j * stride

    if input_array.ndim == 2:
        return input_array[start_i:start_i + filter_height, start_j:start_j + filter_height]
    elif input_array.ndim == 3:
        return input_array[:start_i:start_i + filter_height, start_j:start_j + filter_height]


# 获取一个 2D 区域的最大值的索引
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i = i
                max_j = j
    return max_i, max_j


# 计算卷积
def convolution(input_array, kernel_array, output_array, stride, bias):
    channel_number = input_array.ndim
    output_width = output_array.shape[1]  # 列 -- column
    output_height = output_array.shape[0]  # 行 -- row
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]

    for i in range(output_height):
        for j in range(output_width):
            output_array[i, j] = (get_patch(input_array, i, j, kernel_width, kernel_height,
                                            stride) * kernel_array).sum() + bias


# zero padding
def padding(input_array, zp):
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((input_depth, input_height + 2 * zp, input_width + 2 * zp))
            padded_array[:, zp:zp + input_height, zp:zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((input_height + 2 * zp, input_width + 2 * zp))
            padded_array[zp:zp + input_height, zp:zp + input_width] = input_array
            return padded_array


# not understand
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:%s\n bias:%s' % (repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ConvolutionLayer(object):
    def __init__(self, input_width, input_height, channel_number, filter_height, filter_width, filter_number,
                 zero_padding, stride, activator, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_number = filter_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.zero_padding = zero_padding
        self.stride = stride
        self.learning_rate = learning_rate

        self.output_width = ConvolutionLayer.calculater_output_size(self.input_width, filter_width, zero_padding,
                                                                    stride)
        self.output_height = ConvolutionLayer.calculater_output_size(self.input_height, filter_height, zero_padding,
                                                                     stride)
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.activator = activator
        self.filters = []
        for i in range(self.filter_number):
            self.filters.append(Filter(filter_width, filter_height, self.channel_number))

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2 * zero_padding) // stride + 1

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    def update(self):
        for f in self.filters:
            f.update(self.learning_rate)  # 参数更新

    def forward(self, input_array):
        """
        # 计算卷积层的输出
        # 将结果保存在 self.output_array
        :param input_array: input data
        :return:
        """
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)

        for f in range(self.filter_number):
            filter_ = self.filters[f]
            convolution(self.padded_input_array, filter_.get_weights(), self.output_array[f]
                        , self.stride, filter_.get_bias())

        # activator
        element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        """
        # 计算传递给前一层的 误差项， 以及计算每个权重的梯度
        # 前一层的误差项保存在， self.delta_array
        # 梯度保存在 Filter对象的 weights_grad
        :param input_array:
        :param sensitivity_array:
        :param activator:
        :return:
        """
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        pass

    def bp_gradient(self, sensitivity_array):
        pass


if __name__ == '__main__':
    pass
