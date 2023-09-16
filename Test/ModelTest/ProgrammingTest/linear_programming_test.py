""" 线性规划测试 """
import numpy as np
from Models.programming import LpHelper
import matplotlib.pyplot as plt
from Plot.styles import mp_seaborn_light

e = LpHelper(c=[-1, 2],
             a_ub=[[4, 3], [2, 1]],
             b_ub=[35, 15],
             a_eq=[[1, 1]],
             b_eq=[10],
             bound=[[1, np.inf], [1, np.inf]])

print('线性规划方程:')
e.print_equations()

print("目标函数最小值:", e.res.fun)
print('最优解:', e.res.x)

plt.style.use(mp_seaborn_light())
e.plot_lp()
plt.show()





