""" 线性回归测试 """
from matplotlib import pyplot as plt
from Models.regression import LinearRegression
from Plot.styles import mp_seaborn_light

# 定义数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [2, 4.01, 6.1, 8, 10, 12.1, 14]

# 线性回归
lin = LinearRegression()
lin.fit(x, y)
print('回归方程：', lin.eq)

# 预测
x = [8, 9, 10, 11, 12]
print('预测值：', lin.predict(x))

# 绘图
plt.style.use(mp_seaborn_light())
lin.plot_lr()
lin.plot_residuals()
plt.show()
