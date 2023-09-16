""" 预测模型 """
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.stats.diagnostic import acorr_ljungbox
from Plot.functions import plot_forecast, plot_fitting


class ArimaForcast:
    def __init__(self):
        """ ARIMA 预测 """
        self.data = None
        self.length = None
        self.best_pdq = None
        self.is_stable = None
        self.not_white_noise = None
        self.fit_value = None
        self.forecast = None

    def find_order(self, data, n=3):
        """
        自动寻找最佳的p、d、q值
        :param data: 时间数据序列
        :param n: 最大迭代阶数范围
        :return: 最佳p、d、q值
        """
        p = d = q = range(0, n)
        pdq = list(itertools.product(p, d, q))

        best_aic = np.inf
        best_pdq = None

        for param in pdq:
            model = ARIMA(data, order=param)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param

        self.best_pdq = best_pdq

    def fit(self, data, length: int, pdq: tuple, is_summary=True):
        """
        简单ARIMA预测

        :param data: 原始数据序列
        :param length: 预测步长
        :param pdq: 自回归阶数(p)、差分阶数(d)和移动平均阶数(q)
        :param is_summary: 是否打印总结
        :return:
        """
        adf_result = np.asarray(adf(data))

        # 平稳性检验
        is_stable = adf_result[1] < 0.05

        # 白噪音检验
        not_white_noise = np.all(acorr_ljungbox(data)['lb_pvalue'] < 0.05)

        # ARIMA模型
        model = ARIMA(data, order=pdq)
        fit_value = model.fit()
        if is_summary:
            print(fit_value.summary())
        forecast = fit_value.forecast(length)

        # 储存结果
        self.data = data
        self.length = length
        self.forecast = forecast
        self.is_stable = is_stable
        self.not_white_noise = not_white_noise
        self.fit_value = fit_value

    def plot_fitting(self,
                     title='ARIMA 拟合图',
                     x_label='时间',
                     y_label='y',
                     labels=('原数据', '拟合值'),
                     x_rotation=0,
                     x_ticks_labels=None):
        """
        绘制拟合图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :param labels: 图例
        :param x_rotation: x轴旋转
        :param x_ticks_labels: x轴刻度标签
        :return:
        """
        n = len(self.data)
        x = np.arange(0, n)
        fit_value = self.fit_value.predict(0, n - 1)
        plot_fitting(x, self.data, x, fit_value, title, x_label, y_label, labels, x_rotation, x_ticks_labels)

    def plot_forecast(self,
                      title='ARIMA 预测图',
                      x_label='时间',
                      y_label='y',
                      labels=('原数据', '预测值', '精度误差'),
                      x_rotation=0,
                      x_ticks_labels=None):
        """
        绘制预测图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :param labels: 图例
        :param x_rotation: x刻度旋转角度
        :param x_ticks_labels: x轴刻度标签
        :return:
        """
        plot_forecast(self.data, self.forecast, title, x_label, y_label, labels, x_rotation, x_ticks_labels)


class GM11:
    def __init__(self):
        self.fitting = None
        self.forecast = None
        self.residual = None
        self.data = None

    @staticmethod
    def _fit(data, length: int):
        """
        灰色预测模型GM(1,1)主函数
        :param data: 原始数据序列
        :param length: 预测步长
        :return: 预测值
        """
        data = np.asarray(data)
        n = data.shape[0]

        # 累加序列
        cum = np.cumsum(data)
        cum_mean = (cum[:-1] + cum[1:]) / 2
        cum_data = np.vstack([-cum_mean, np.ones_like(cum_mean)]).T

        # 最小二乘法求解系数a和b
        a, b = np.linalg.lstsq(cum_data, data[1:], rcond=None)[0]

        # 白化方程
        result = (data[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * np.arange(0, n + length))
        return result

    def fit(self, data, length):
        """
        灰色预测模型主函数
        :param data: 数据
        :param length: 预测步长
        :return:
        """
        self.data = data.copy()

        # 级比平移
        deta = 0
        while not self.grade_ratio(data):
            deta = np.max(data)
            data += deta

        result = self._fit(data, length)
        result -= deta

        fitting, forcast = np.split(result, [data.shape[0]])
        residual = data - deta - fitting

        self.fitting = fitting
        self.forecast = forcast
        self.residual = residual

    @staticmethod
    def grade_ratio(data, r_tol=1e-8):
        """
        级比检验
        :param data: N,M 特征序列
        :param r_tol: 微小值
        :return: 是否通过检验
        """
        data = np.where(data == 0, r_tol, data)

        # 计算级比
        diff = np.diff(data)
        ratio = 1 - np.divide(diff, data[:-1])

        # 计算最小和最大的阈值
        n = data.shape[0]
        min_ = np.e ** (-2 / (n + 1))
        max_ = np.e ** (2 / n + 1)

        # 判断级比是否在阈值范围内
        is_pass = np.all(ratio > min_) & np.all(ratio < max_)
        return is_pass

    def posterior_error(self):
        """
        后验差检验
        :return: 检验结果
        """
        actual = self.data
        residual = self.residual

        c = np.std(residual) / np.std(actual)
        if c < 0.35:
            eva = '优秀'
        elif c < 0.5:
            eva = '合格'
        elif c < 0.65:
            eva = '一般'
        else:
            eva = '不合格'
        df = pd.DataFrame({'C值': c, '评价': eva}, index=['后验差检验'])
        return df

    def small_error(self):
        """
        小误差概率检验
        :return:
        """
        actual = np.asarray(self.data)
        residual = self.residual

        # 计算小误差概率P值
        var = np.var(actual)
        sv = np.sqrt(var)
        sr = np.abs(residual - np.mean(residual))
        p = np.extract(sr < norm.ppf(0.75) * sv, sr).shape[0] / actual.shape[0]

        if p > 0.95:
            eva = '优秀'
        elif p > 0.8:
            eva = '合格'
        elif p > 0.7:
            eva = '一般'
        else:
            eva = '不合格'
        df = pd.DataFrame({'P值': p, '评价': eva}, index=['小误差概率检验'])
        return df

    def plot_fitting(self,
                     title='灰色预测拟合图',
                     x_label='时间',
                     y_label='y',
                     labels=('原数据', '拟合值'),
                     x_rotation=0,
                     x_ticks_labels=None):
        """
        绘制拟合图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :param labels: 图例
        :param x_rotation: x轴旋转
        :param x_ticks_labels: x轴刻度标签
        :return:
        """
        x = np.arange(0, len(self.data))
        plot_fitting(x, self.data, x, self.fitting, title, x_label, y_label, labels, x_rotation, x_ticks_labels)

    def plot_forecast(self,
                      title='灰色预测图',
                      x_label='x',
                      y_label='y',
                      labels=('原数据', '预测值', '精度误差'),
                      x_rotation=0,
                      x_ticks_labels=None):
        """
        绘制预测图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :param labels: 图例
        :param x_rotation: x轴旋转
        :param x_ticks_labels: x轴刻度标签
        :return:
        """
        plot_forecast(self.data, self.forecast, title, x_label, y_label, labels, x_rotation, x_ticks_labels)


def markov_chain(tran_prob, states, init_state, length=10, random_seed=0):
    """
    MarkovChain 马尔可夫链预测模型

    :param tran_prob: 状态转移概率列表，表示从当前状态到其他状态的概率。
    :param states: 状态列表，包含所有可能的状态。
    :param init_state: 初始状态
    :param length: 预测长度
    :param random_seed: 随机种子
    :return: 生成器对象，每次调用该函数返回下一个状态
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    current_idx = states.index(init_state)

    forecast = []
    n_states = len(states)
    for _ in range(length):
        prob = tran_prob[current_idx]
        next_idx = np.random.choice(n_states, p=prob)
        current_idx = next_idx
        current_state = states[next_idx]
        forecast.append(current_state)

    return forecast

