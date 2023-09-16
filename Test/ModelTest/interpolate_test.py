""" 插值测试 """
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from Plot.styles import mp_seaborn_light

# 创建数据
x = np.arange(0, 10)
y = np.exp(x)
x_new = np.linspace(0, 9, 20)

# 进行线性插值
f_linear = interpolate.interp1d(x, y)
y_linear = f_linear(x_new)

# 进行样条插值
f_spline = interpolate.splrep(x, y)
y_spline = interpolate.splev(x_new, f_spline)
"""
f_spline = interpolate.interp1d(x, y, kind='cubic')
y_spline = f_spline(x_new)
"""

# 进行最近邻插值
f_nearest = interpolate.interp1d(x, y, kind='nearest')
y_nearest = f_nearest(x_new)
"""
y_nearest = np.zeros_like(x_new)
for i, xn in enumerate(x_new):
    idx = np.abs(x - xn).argmin()
    y_nearest[i] = y[idx]
"""

# 创建拉格朗日插值函数
f_lagrange = lagrange(x, y)
y_lagrange = f_lagrange(x_new)


def lagrange_inter(x, y, x_inter):
    """
    拉格朗日插值法
    :param x: 已知点的x坐标列表
    :param y: 已知点的y坐标列表
    :param x_inter: 用于插值的x坐标
    :return: 插值点的y坐标
    """
    n = len(x)
    y_inter = 0
    for i in range(n):
        # 拉格朗日基函数
        lag = 1
        for j in range(n):
            if i != j:
                lag *= (x_inter - x[j]) / (x[i] - x[j])
        y_inter += y[i] * lag
    return y_inter


# print(lagrange_interpolation(x, y, x_new))


# 输出结果
print("插值后的函数值（线性插值）：", y_linear)
print("插值后的函数值（样条插值）：", y_spline)
print("插值后的函数值（最近邻插值）：", y_nearest)
print("插值后的函数值（拉格朗日插值）：", y_lagrange)

plt.style.use(mp_seaborn_light())
plt.figure(figsize=(14, 8))

plt.subplot(231)
plt.plot(x, y, "o", label='原数据', markeredgecolor="white")
plt.legend()

plt.subplot(232)
plt.plot(x_new, y_linear, 'o--', label='线性插值', markeredgecolor="white")
plt.legend()

plt.subplot(233)
plt.plot(x_new, y_spline, 'o--', label='样条插值', markeredgecolor="white")
plt.legend()

plt.subplot(234)
plt.plot(x_new, y_nearest, 'o--', label='最近邻插值', markeredgecolor="white")
plt.legend()

plt.subplot(235)
plt.plot(x_new, y_lagrange, 'o--', label='拉格朗日插值', markeredgecolor="white")
plt.legend()

plt.show()

# pandas 同样提供了插值相关函数

# 创建DataFrame
df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5]})

# 线性插值
df_linear = df.interpolate(method='linear')

# 三次样条插值
df_spline = df.interpolate(method='spline', order=3)

# 最近邻插值
df_nearest = df.interpolate(method='nearest')

print("插值后的函数值（线性插值）：", df_linear)
print("插值后的函数值（样条插值）：", df_spline)
print("插值后的函数值（最近邻插值）：", df_nearest)
