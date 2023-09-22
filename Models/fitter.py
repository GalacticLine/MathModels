""" 数据拟合 """
import sys
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
from Plot.functions import plot_residuals, plot_fitting


def sigmoid(x, a, b, c):
    """ 逻辑函数 """
    return a / (1 + np.exp(-b * (x - c)))


def sp_sigmoid():
    """ 逻辑函数 sympy """
    x, a, b, c = sp.symbols('x a b c')
    expr = a / (1 + sp.exp(-b * (x - c)))
    return (a, b, c), expr


def gaussian(x, a, b, c):
    """ 高斯函数 """
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def sp_gaussian():
    """ 高斯函数 sympy """
    x, a, b, c = sp.symbols('x a b c')
    expr = a * sp.exp(-(x - b) ** 2 / (2 * c ** 2))
    return (a, b, c,), expr


def exponential(x, a, b, c):
    """ 指数函数 """
    return a * np.exp(b * x) + c


def sp_exponential():
    """ 指数函数 sympy """
    x, a, b, c = sp.symbols('x a b c')
    expr = a * sp.exp(b * x) + c
    return (a, b, c), expr


def logarithmic(x, a, b, c):
    """ 对数函数 """
    return a * np.log(b * x) + c


def sp_logarithmic():
    """ 对数函数 sympy """
    x, a, b, c = sp.symbols('x a b c')
    expr = a * sp.log(b * x) + c
    return (a, b, c), expr


def get_fit_func(name, is_sympy=False):
    """
    获取当前模块下的拟合函数
    :param name: 拟合函数名
    :param is_sympy: 是否获取sympy版函数
    :return: 函数对象
    """
    if is_sympy:
        name = f'sp_{name}'
    return getattr(sys.modules[__name__], name)


class Fitter:
    def __init__(self):
        """
        拟合工具箱
        """
        self.x = None
        self.y = None
        self.mode = None
        self.func = None
        self.formula = None
        self.coefficients = None

    def poly_fit(self, x, y, deg=2, precision=2):
        """
        多项式拟合
        :param x: 自变量x
        :param y: 因变量y
        :param deg: 多项式最高阶数，当为1时，相当于线性拟合
        :param precision: 精度
        :return:
        """
        # 系数
        coefficients = np.polyfit(x, y, deg)
        coefficients = np.round(coefficients, precision)

        # 方程
        poly = np.poly1d(coefficients)

        # sympy方程
        x_sp, y_sp = sp.symbols('x y')
        formula = sp.expand(sp.Eq(y_sp, poly(x_sp)))

        # 储存结果
        self.x = x
        self.y = y
        self.mode = 'linear'
        self.func = poly
        self.formula = formula
        self.coefficients = coefficients

    def curve_fit(self, x, y, name='sigmoid', precision=3):
        """
        曲线拟合

        :param x: 自变量
        :param y: 因变量
        :param name: 拟合函数名
        :param precision: 精度
        :return:
        """
        # 方程
        func = get_fit_func(name)

        # 系数
        coefficients = curve_fit(func, x, y)[0]

        # sympy方程
        sp_func = get_fit_func(name, is_sympy=True)
        sp_args, sp_formula = sp_func()
        sp_formula = sp.Eq(sp.symbols('y'), sp_formula)
        sp_popt = [sp.N(coe, precision) for coe in coefficients]
        results = [(val, sp_popt[i]) for i, val in enumerate(sp_args)]
        formula = sp_formula.subs(results)

        # 储存结果
        self.x = x
        self.y = y
        self.mode = 'curve'
        self.func = func
        self.formula = formula
        self.coefficients = coefficients

    def plot_fit(self, title='拟合图像', x_label='x', y_label='y'):
        """
        绘制拟合图像
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :return:
        """
        x = self.x
        func = self.func

        x_fit = np.linspace(min(x), max(x), 100)
        if self.mode == 'linear':
            y_fit = func(x_fit)
        else:
            y_fit = func(x_fit, *self.coefficients)

        plot_fitting(x, self.y, x_fit, y_fit, title, x_label, y_label, latex=sp.latex(self.formula))

    def plot_res(self, alpha=0.95, title='残差图', x_label='y'):
        """
        绘制残差图
        :param alpha: 置信区间
        :param title: 标题
        :param x_label: x轴标题
        :return:
        """
        x = self.x
        y = self.y
        func = self.func
        if self.mode == 'linear':
            y_pre = func(x)
        else:
            y_pre = func(x, *self.coefficients)
        plot_residuals(y, y_pre, title, x_label, alpha)
