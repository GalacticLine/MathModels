""" 神经网络模型 """
import sys
import numpy as np
from matplotlib import pyplot as plt
from Plot.functions import plot_network


def sigmoid(x):
    """
    Sigmoid 函数
    :param x: 自变量x
    :return: 因变量y
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    TanH 函数
    :param x: 自变量x
    :return: 因变量y
    """
    return np.tanh(x)


def relu(x):
    """
    ReLU 函数
    :param x: 自变量x
    :return: 因变量y
    """
    return np.maximum(0, x)


def elu(x, alpha=1.0):
    """
    ELU 函数
    :param x: 自变量x
    :param alpha: 幅度
    :return: 因变量y
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def leaky_relu(x, leaky=0.1):
    """
    LeakyReLU 函数
    :param x: 自变量x
    :param leaky: 小斜率
    :return: 因变量y
    """
    return np.maximum(leaky * x, x)


def d_sigmoid(x):
    """
    Sigmoid 导函数
    :param x: 自变量x
    :return: 因变量y
    """
    return sigmoid(x) * (1 - sigmoid(x))


def d_tanh(x):
    """
    TanH 导函数
    :param x: 自变量x
    :return: 因变量y
    """
    return 1 - np.tanh(x) ** 2


def d_relu(x):
    """
    ReLU 导函数
    :param x: 自变量x
    :return: 因变量y
    """
    return np.where(x > 0, 1, 0)


def d_leaky_relu(x, leaky=0.1):
    """
    LeakyReLU导函数

    :param x: 自变量x
    :param leaky: 小斜率
    :return: 因变量y
    """
    return np.where(x > 0, 1, leaky)


def d_elu(x, alpha=1.0):
    """
    ELU 导函数
    :param x: 自变量x
    :param alpha: 幅度
    :return: 因变量y
    """
    return np.where(x > 0, 1, alpha * np.exp(x))


def get_active_func(name, derivative=False):
    """
    获取当前模块下的激活函数
    :param name: 函数名
    :param derivative: 是否为对应导函数
    :return: 函数对象
    """
    if derivative:
        name = f'd_{name}'
    return getattr(sys.modules[__name__], name)


class BPNet:
    def __init__(self,
                 input_size: int,
                 hidden_sizes,
                 output_size: int,
                 active_func='sigmoid',
                 init_func='he'):
        """
        BP神经网络实现。
        :param input_size: 输入层层数
        :param hidden_sizes: 隐藏层层数集合
        :param output_size: 输出层层数
        :param active_func: 激活函数
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.weights, self.biases = self.__init(init_func)

        self.active = get_active_func(active_func)
        self.d_active = get_active_func(active_func, True)

        self.hidden_outputs = [None] * len(hidden_sizes)
        self.outputs = None

        self.weight_delta = [0] * (len(hidden_sizes) + 1)
        self.bias_delta = [0] * (len(hidden_sizes) + 1)

    def __init(self, func_name):
        """
        Xavier 初始化函数，用于初始化神经网络的权重和偏置。
        :return: 权重和偏置
        """

        output_size = self.output_size
        prev_size = self.input_size
        hidden_sizes = self.hidden_sizes

        weights = []
        biases = []

        def xavier(size):
            limit = np.sqrt(6 / (prev_size + size))
            weight = np.random.uniform(-limit, limit, (prev_size, size))
            weights.append(weight)
            biases.append(np.zeros((1, size)))

        def he(size):
            limit = np.sqrt(2 / prev_size)
            weight = np.random.uniform(-limit, limit, (prev_size, size))
            weights.append(weight)
            biases.append(np.zeros((1, size)))

        if func_name == 'xavier':
            func = xavier
        elif func_name == 'he':
            func = he
        else:
            raise ValueError(f"没有名为{func_name}的初始化函数。")

        for hidden_size in hidden_sizes:
            func(hidden_size)
            prev_size = hidden_size
        func(output_size)

        return weights, biases

    def _forward(self, x):
        weights = self.weights
        biases = self.biases
        func = self.active

        xs = []
        for i in range(len(self.hidden_sizes)):
            x = func(x @ weights[i] + biases[i])
            xs.append(x)
        output = func(x @ weights[-1] + biases[-1])
        return xs, output

    def forward(self, x):
        """
        前馈
        :param x: 自变量x
        :return: 输出
        """
        self.hidden_outputs, self.outputs = self._forward(x)

    def backward(self, x, y, lr, momentum, mode, lamb=0):
        """
        后馈
        :param x: 自变量x
        :param y: 因变量y
        :param lr: 学习率
        :param momentum: 动量因子
        :param mode: 正则化模式
        :param lamb: 正则化的系数
        :return:
        """
        func = self.d_active
        weights = self.weights
        outputs = self.outputs
        hidden_outputs = self.hidden_outputs
        n = len(self.hidden_sizes)

        hidden_delta = [0] * n
        hidden_delta.append((y - outputs) * func(outputs))

        for i in reversed(range(n)):
            hidden_delta[i] = np.dot(hidden_delta[i + 1], weights[i + 1].T) * func(hidden_outputs[i])

        for i in reversed(range(n + 1)):
            if i == 0:
                data = x
            else:
                data = hidden_outputs[i - 1]
            if mode == 'L1':
                lamb_weight = - lamb * np.sign(weights[i])
            else:
                lamb_weight = - lamb * weights[i]

            weight_grad = data.T @ hidden_delta[i] + lamb_weight
            bias_grad = np.sum(hidden_delta[i], axis=0)

            self.weight_delta[i] = momentum * self.weight_delta[i] + lr * weight_grad
            self.bias_delta[i] = momentum * self.bias_delta[i] + lr * bias_grad
            self.weights[i] += self.weight_delta[i]
            self.biases[i] += self.bias_delta[i]

    def train(self, x_train, y_train, epochs=1000, lr=0.1, momentum=0.9, mode='L2', lamb=0):
        """
        训练
        :param x_train: 自变量x
        :param y_train: 因变量y
        :param epochs: 迭代次数
        :param lr: 学习率
        :param momentum: 动量因子
        :param mode: 正则化模式
        :param lamb: 正则化的系数
        :return:
        """
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        for i in range(epochs):
            self.forward(x_train)
            self.backward(x_train, y_train, lr, momentum, mode, lamb)
            loss = np.mean(np.square(y_train - self.outputs))

            print(f'[{i + 1}/{epochs}]  loss: {loss}')

    def predict(self, x_test):
        """
        预测
        :param x_test: 自变量x
        :return: 预测结果
        """
        return self._forward(x_test)[-1]

    def evaluate(self, x_test, y_test, decimals=2):
        """
        评估模型
        :param x_test: 自变量x
        :param y_test: 因变量x
        :param decimals: 精度
        :return: r方
        """
        pre = self._forward(x_test)[-1]

        residuals = np.round(y_test - pre, decimals)
        ss_residuals = np.sum(np.square(residuals))

        ss_total = np.sum(np.square(y_test - np.mean(y_test)))

        r_squared = 1 - (ss_residuals / (ss_total + 1e-8))
        r_squared = f"{r_squared * 100:.2f}%"
        return r_squared

    def plot_data(self, x, y, x_label='x', y_label='y'):
        """
        :param x: 自变量x
        :param y: 因变量y
        :param x_label: x轴标题
        :param y_label: y轴标题
        :return:
        """
        y_pre = self.predict(x)

        plt.figure()
        plt.plot(x, y, label='原数据')
        plt.plot(x, y_pre, label='预测值')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    def plot_network(self):
        """
        绘制神经网络结构图
        :return:
        """
        plot_network(self.input_size, self.hidden_sizes, self.output_size)
