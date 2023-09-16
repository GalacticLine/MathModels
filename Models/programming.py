""" 规划模型 """
import itertools
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog
from sympy.printing import latex


class LpHelper:
    def __init__(self, *, c, a_ub=None, b_ub=None, a_eq=None, b_eq=None, bound=None):
        """
        线性规划辅助类。
        :param c: 目标函数的系数
        :param a_ub: 不等式约束条件的系数
        :param b_ub: 不等式约束条件的右侧常数
        :param a_eq: 等式约束条件的系数
        :param b_eq: 等式约束条件的右侧常数
        :param bound: 变量的上下界
        """
        self.c = c
        self.a_ub = a_ub
        self.b_ub = b_ub
        self.a_eq = a_eq
        self.b_eq = b_eq

        if bound is None:
            bound = [[-np.inf, np.inf]] * len(c)
        else:
            lack = len(c) - len(bound)
            if lack != 0:
                for _ in range(lack):
                    bound.append([-np.inf, np.inf])

        self.bound = bound
        self.res = linprog(c, a_ub, b_ub, a_eq, b_eq, bound)
        self._equations = None
        self._points = None

    @property
    def equations(self):
        """
        获取标准型方程组。
        :return: 方程集合
        """
        if self._equations is None:
            xs = sp.symbols([f'x{i}' for i in range(len(self.c))])
            target = np.dot(self.c, xs)

            subjects = []

            if self.a_ub is not None:
                for a, b in zip(self.a_ub, self.b_ub):
                    subjects.append(sp.LessThan(np.dot(a, xs), b))

            if self.a_eq is not None:
                for a, b in zip(self.a_eq, self.b_eq):
                    subjects.append(sp.Eq(np.dot(a, xs), b))

            for idx, x in enumerate(xs):
                subjects.append(sp.GreaterThan(x, self.bound[idx][0]))

            self._equations = target, subjects
        return self._equations

    def print_equations(self):
        """
        打印方程组
        :return:
        """
        z_eq, equations = self.equations
        print('min ', end='')
        sp.pprint(z_eq)
        print('ModernTest.t.')
        for eq in equations:
            sp.pprint(eq)

    def plot_lp(self,
                x_min=-20,
                x_max=20,
                y_min=-20,
                y_max=20,
                zero_spines=True,
                title='线性规划',
                x_label='x',
                y_label='y'):
        """
        绘制线性规划图像。

        :param x_min: x轴最小值
        :param x_max: x轴最大值
        :param y_min: y轴最小值
        :param y_max: y轴最大值
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :param zero_spines: 坐标轴是否以零点为中心
        :return:
        """
        if len(self.c) > 2:
            raise ValueError('不支持二元以上线性规划图像绘制')
        res = self.res

        fig, ax = plt.subplots()
        if zero_spines:
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        x = np.array([x_min, x_max])

        if self.a_ub is not None:
            ab = np.asarray(self.a_ub)
            x0 = (self.b_ub - ab[:, 0] * x[0]) / ab[:, 1]
            x1 = (self.b_ub - ab[:, 0] * x[1]) / ab[:, 1]
            point = np.column_stack((x0, x1))
            plt.plot(x, point.T, linestyle='--')

        if self.a_eq is not None:
            aq = np.asarray(self.a_eq)
            x0 = (self.b_eq - aq[:, 0] * x[0]) / aq[:, 1]
            x1 = (self.b_eq - aq[:, 0] * x[1]) / aq[:, 1]
            point = np.column_stack((x0, x1))
            plt.plot(x, point.T)

        for v2 in self.bound[0]:
            plt.axvline(v2, color='grey', linestyle='--')
        for v2 in self.bound[1]:
            plt.axhline(v2, color='grey', linestyle='--')

        if res.x is not None:
            plt.plot(res.x[0], res.x[1], 'ro', markersize=8, markerfacecolor='w')

        if len(self.points) > 2:
            sorted_arr = sorted(self.points, key=lambda elem: (elem[0], elem[1]))
            polygon = Polygon(sorted_arr)
            ax.add_patch(polygon)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        eqs = [r'$%s$' % latex(eq) for eq in self.equations[1]]
        plt.legend(eqs)

    @property
    def points(self):
        """
        获取可行域轮廓交点。
        :return: 交点坐标
        """
        if self._points is None:
            points = []
            if self.res.x is not None:
                bound = self.bound

                coe = np.vstack((self.a_ub, [[1, 0], [0, 1], [1, 1], [0, 1]]))
                intercept = np.hstack((self.b_ub, np.asarray(bound).flatten(order='F')))

                coe = itertools.combinations(coe, 2)
                intercept = itertools.combinations(intercept, 2)

                for x, y in zip(coe, intercept):
                    if not np.all(np.isinf(y)) and np.linalg.matrix_rank(x) != 1:
                        sol = np.linalg.solve(x, y)
                        if np.all(np.isinf(sol)):
                            continue
                        if not bound[0][1] >= sol[0] >= bound[0][0]:
                            continue
                        if not bound[1][1] >= sol[1] >= bound[1][0]:
                            continue
                        if not np.all(np.round(self.a_ub @ sol, 8) <= self.b_ub):
                            continue
                        points.append(sol)

            self._points = points
        return self._points
