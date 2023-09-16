""" 拟合测试 """
import numpy as np
from matplotlib import pyplot as plt
from Models.fitter import Fitter
from Plot.styles import mp_seaborn_light

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 24]
x = np.array(x)
y = np.array(y)

plt.style.use(mp_seaborn_light())

model = Fitter()

model.poly_fit(x, y, 2)
model.plot_fit(title='二阶多项式拟合')
model.plot_res(title='二阶多项式拟合残差图')
plt.show()

model.curve_fit(x, y, 'sigmoid')
model.plot_fit(title='Sigmoid函数拟合')
model.plot_res(title='Sigmoid函数拟合残差图')
plt.show()
