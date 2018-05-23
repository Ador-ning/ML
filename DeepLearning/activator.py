# coding:utf-8

import numpy as np


class ReLUActivator(object):
    @staticmethod
    def forward(self, weight_input):
        return max(0, weight_input)

    @staticmethod
    def backward(self, output):
        return 1 if output > 0 else 0


class IdentityActivator(object):
    @staticmethod
    def forward(weight_input):
        return weight_input

    @staticmethod
    def backward(output):
        return 1


class SigmoidActivator(object):
    @staticmethod
    def forward(self, weight_input):
        return np.longfloat(1.0 / (1.0 + np.exp(-weight_input)))

    @staticmethod
    def backward(self, output):
        return output * (1 - output)


class TanhActivator(object):
    @staticmethod
    def forward(self, weight_input):
        return 2.0 / (1.0 + np.exp(-2 * weight_input)) - 1.0

    @staticmethod
    def backward(self, output):
        return 1 - output * output


if __name__ == '__main__':
    pass
