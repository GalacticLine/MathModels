""" 微分方程模型 """
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


class SI:
    def __init__(self, *,
                 population=7e9,
                 i=1,
                 beta=0.1,
                 time=365 * 3):
        """
        SI模型
        :param population: 总人口数
        :param i: 感染人数
        :param beta: 感染率
        :param time: 时间（天）
        """
        time = range(0, time)
        s = population - i
        beta_ratio = beta / population
        result = odeint(self.si, [s, i], time, args=(beta_ratio,))
        self.result = np.asarray(result).T
        self.time = time

    @staticmethod
    def si(y, t, beta_ratio):
        """
        SI模型 微分方程实现。
        :param y: (ModernTest, i)
        :param t: 时间
        :param beta_ratio: 感染率与总人口之比
        :return: (ds/dt, di/dt)
        """
        s, i = y
        ds_dt = -beta_ratio * s * i
        di_dt = -ds_dt
        return ds_dt, di_dt

    def plot_model(self, title='SI模型', x_label='时间(天)', y_label='人数'):
        """
        绘制SI模型图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :return:
        """
        labels = ['易感者', '感染者']

        plt.figure()
        for label, data in zip(labels, self.result):
            plt.plot(self.time, data, label=label)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()


class SIR:
    def __init__(self, *,
                 population=7e9,
                 i=1,
                 r=0,
                 beta=0.1,
                 gamma=0.01,
                 time=365 * 3):
        """
        SIR模型
        :param population: 总人口数
        :param i: 感染人数
        :param r: 康复人数
        :param beta: 感染率
        :param gamma: 恢复率
        :param time: 时间（天）
        """

        time = range(0, time)
        s = population - i - r
        beta_ratio = beta / population
        result = odeint(self.sir, [s, i, r], time, args=(beta_ratio, gamma))
        self.result = np.asarray(result).T
        self.time = time

    @staticmethod
    def sir(y, t, beta_ratio, gamma):
        """
        SI模型 微分方程实现。
        :param y: (ModernTest, i, r)
        :param t: 时间
        :param beta_ratio: 感染率与总人口之比
        :param gamma: 恢复率
        :return: (ds/dt, di/dt, dr/dt)
        """
        s, i, r = y
        ds_dt = -beta_ratio * s * i
        dr_dt = gamma * i
        di_dt = -ds_dt - dr_dt
        return ds_dt, di_dt, dr_dt

    def plot_model(self, title='SIR模型', x_label='时间(天)', y_label='人数'):
        """
        绘制SIR模型图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :return:
        """
        labels = ['易感者', '感染者', '康复者']

        plt.figure()
        for label, data in zip(labels, self.result):
            plt.plot(self.time, data, label=label)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()


class SEIRD:
    def __init__(self, *,
                 population=7e9,
                 e=0, i=1, r=0, d=0,
                 beta=0.3,
                 gamma=0.1,
                 alpha=0.1,
                 delta=0.03,
                 time=365 * 3):
        """
        模型开始模拟。
        :param population: 总人口数
        :param e: 潜伏人数
        :param i: 感染人数
        :param r: 康复人数
        :param d: 死亡人数
        :param beta: 感染率
        :param gamma: 恢复率
        :param alpha: 潜伏期转化率
        :param delta: 死亡率
        :param time: 时间（天）
        """

        time = range(0, time)
        beta_ratio = beta / population

        init = [e, i, r, d]
        init.insert(0, population - np.sum(init))
        args = alpha, beta_ratio, gamma, delta

        result = odeint(self.seird, init, time, args=args)

        self.result = np.round(np.array(result).T)
        self.time = time

    @staticmethod
    def seird(y, t, alpha, beta_ratio, gamma, delta):
        """
        SEIRD模型 微分方程实现。
        :param y: (ModernTest, e, i, r, d)
        :param t: 时间
        :param alpha: 潜伏期转化率
        :param beta_ratio: 感染率与总人口之比
        :param gamma: 恢复率
        :param delta: 死亡率
        :return: (ds/dt, de/dt, di/dt, dr/dt, dd/dt)
        """
        s, e, i, r, d = y
        delta_num = alpha * e
        ds_dt = - beta_ratio * s * i
        de_dt = - ds_dt - delta_num
        dr_dt = gamma * i
        dd_dt = delta * i
        di_dt = delta_num - dr_dt - dd_dt
        result = ds_dt, de_dt, di_dt, dr_dt, dd_dt
        return result

    def plot_model(self, title='SEIRD模型', x_label='时间(天)', y_label='人数'):
        """
        绘制SEIRD模型图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :return:
        """
        labels = ['易感者', '潜伏者', '感染者', '康复者', '死亡者']

        plt.figure()
        for label, data in zip(labels, self.result):
            plt.plot(self.time, data, label=label)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()


class LogisticPopulation:
    def __init__(self, rate=0.03, volume=1000, population=100, time=365 * 3):
        """
        Logistic 人口增长模型

        :param rate: 自然增长率
        :param volume: 环境容量
        :param population: 初始人口
        :param time: 时间
        """
        time = range(0, time)

        self.time = time
        self.population = odeint(self.population_growth, population, time, args=(rate, volume))

    @staticmethod
    def population_growth(y, t, r, K):
        """
        Logistic 人口增长模型 微分方程实现。
        :param y: 人口
        :param t: 时间
        :param r: 自然增长率
        :param K: 环境容量
        :return: dy/dt
        """
        dy_dt = r * y * (1 - y / K)
        return dy_dt

    def plot_model(self, title='人口增长模型', x_label='时间(天)', y_label='人数'):
        """
        绘制人口增长模型图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :return:
        """

        plt.figure()
        plt.plot(self.time, self.population)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)


class Lorenz:
    def __init__(self,
                 init_xyz=(1.0, 1.0, 1.0),
                 sigma=10.0,
                 beta=8.0 / 3.0,
                 rho=28.0,
                 time=20,
                 max_step=0.01):
        """
        洛伦兹方程
        :param init_xyz: 初始坐标
        :param sigma:  x 方向上的耗散率
        :param beta: y 方向上的耗散率和 x、z 之间的耦合
        :param rho: z 方向上的耗散率和 x、y 之间的正向反馈
        :param time: 时间
        :param max_step: 积分的最大步长
        """
        solution = solve_ivp(self.lorenz_equations, [0, time], init_xyz, args=(sigma, beta, rho), max_step=max_step)
        self.solution = solution

    @staticmethod
    def lorenz_equations(t, s, sigma, beta, rho):
        """
        洛伦兹方程的微分方程实现

        :param t: 时间
        :param s: (x, y, z)
        :param sigma: x 方向上的耗散率
        :param beta: y 方向上的耗散率和 x、z 之间的耦合
        :param rho: z 方向上的耗散率和 x、y 之间的正向反馈
        :return: (dx/dt, dy/dt, dz/dt)
        """
        x, y, z = s
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]

    def plot_model(self, title='洛伦兹方程'):
        """
        绘制洛伦兹方程图像
        :param title: 标题

        :return:
        """
        xyz = getattr(self.solution, 'y')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*xyz)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
