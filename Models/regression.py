""" 回归模型 """
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from Plot.functions import plot_residuals, plot_fitting


class LinearRegression:

    def __init__(self):
        """
        线性回归
        """
        self.x = None
        self.y = None
        self.args = None

    def fit(self, x, y):
        """
        线性回归
        最小二乘法进行线性回归的参数估计，返回拟合直线的截距和斜率作为结果
        :param x: 自变量
        :param y: 因变量
        :return: 回归结果
        """
        self.x = x
        self.y = y
        x = np.reshape(x, -1)
        x = np.vstack((np.ones(x.size), x)).T

        self.args = np.linalg.lstsq(x, y, rcond=None)[0]

    def predict(self, x):
        """
        预测
        :param x: 自变量x
        :return: 预测结果
        """
        x = np.reshape(x, -1)
        x = np.vstack((np.ones(x.size), x)).T

        y_pre = x @ self.args
        return y_pre

    @property
    def eq(self, precision=2):
        """
        获取方程
        :return: 方程
        """
        intercept, coe = np.round(self.args, precision)

        x, y = sp.symbols('x y')
        eq = sp.Eq(y, coe * x + intercept)
        eq_pretty = sp.pretty(eq)

        return eq_pretty

    def plot_lr(self, title='线性回归图', x_label='x', y_label='y', labels=('原数据', '回归值')):
        x = self.x
        plot_fitting(x, self.y, x, self.predict(x), title, x_label, y_label, labels, latex=self.eq)

    def plot_residuals(self, r_title='残差图', x_label='y', alpha=0.95):
        plot_residuals(self.y, self.predict(self.x), r_title, x_label, alpha)


class LogisticRegression:

    def __init__(self):
        """
        逻辑回归类

        """
        self._x = None
        self._y = None
        self.bias = None
        self.weights = None

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid 函数
        :param x: 自变量x
        :return: 因变量y
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y, lr=1.0, n_iter=100):
        """
        拟合
        :param x: 自变量x
        :param y: 因变量y
        :param lr: 学习率
        :param n_iter: 迭代数
        :return:
        """
        x = np.asarray(x)
        y = np.asarray(y)

        n_samples, n_features = x.shape
        weights = np.zeros(n_features)
        bias = 0
        for _ in range(n_iter):
            model = np.dot(x, weights) + bias
            res = self.sigmoid(model) - y
            n = 1 / n_samples
            dw = n * np.dot(x.T, res)
            db = n * np.sum(res)
            weights -= lr * dw
            bias -= lr * db

        self._x = x
        self._y = y
        self.bias = bias
        self.weights = weights

    def predict(self, x):
        """
        预测
        :param x: 自变量x
        :return: 预测结果
        """
        model = np.dot(x, self.weights) + self.bias
        y_pre = self.sigmoid(model)
        y_pre = np.round(y_pre)
        return y_pre

    def plot_logist_r(self, title='逻辑回归图', x_label='x', y_label='y', pad=0.5):
        """
        绘制分类结果和决策边界
        """
        x = self._x
        y = self._y
        bias = self.bias
        weights = self.weights

        x0_min = np.min(x[:, 0])
        x0_max = np.max(x[:, 0])
        y0_min = np.min(x[:, 1])
        y0_max = np.max(x[:, 1])

        x_values = np.array([x0_min, x0_max])
        y_values = -(bias + weights[0] * x_values) / weights[1]

        plt.scatter(x[y == 0, 0], x[y == 0, 1], s=50, label='0')
        plt.scatter(x[y == 1, 0], x[y == 1, 1], s=50, label='1')

        plt.xlim(x0_min - pad, x0_max + pad)
        plt.ylim(y0_min - pad, y0_max + pad)

        plt.plot(x_values, y_values, '--', label='决策边界', c='#9ec3db')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()


class RidgeRegression:
    def __init__(self):
        """
        岭回归

        """
        self._x = None
        self._y = None
        self.alpha = None
        self.coefficients = None

    @staticmethod
    def _fit(x, y, alpha=0.1):
        x = np.asarray(x)
        y = np.asarray(y)
        new_x = np.hstack([np.ones((x.shape[0], 1)), x])
        ridge = alpha * np.identity(new_x.shape[1])
        inv = np.linalg.inv(new_x.T @ new_x + ridge)
        coefficients = inv @ new_x.T @ y

        return coefficients

    def fit(self, x, y, alpha=0.1):
        """
        拟合
        :param x: 自变量x
        :param y: 因变量y
        :param alpha: 正则化参数
        :return:
        """
        coefficients = self._fit(x, y, alpha)
        self._x = x
        self._y = y
        self.coefficients = coefficients
        self.alpha = alpha

    def predict(self, x):
        """
        预测
        :param x: 自变量x
        :return: 预测结果
        """
        x = np.asarray(x)
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        y_pre = x @ self.coefficients
        return y_pre

    def plot_coefficients(self, alphas=None):
        """
        绘制不同 alpha 下模型的回归系数
        """
        if alphas is None:
            alphas = np.logspace(-4, 5, num=100)

        all_coefficients = []
        for alpha in alphas:
            coe_s = self._fit(self._x, self._y, alpha=alpha)
            all_coefficients.append(coe_s[1:])
        plt.figure()
        plt.plot(alphas, all_coefficients)
        plt.hlines(0, np.min(alphas), np.max(alphas), colors='#666666', linestyles='dashed')
        plt.xscale('log')
        plt.xlabel('正则化参数')
        plt.ylabel('岭系数')
        plt.title('岭系数与正则化')
