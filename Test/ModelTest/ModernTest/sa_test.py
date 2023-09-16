""" 退火算法测试 """
import numpy as np
from matplotlib import pyplot as plt
from Models.simulated_annealing import SimulatedAnnealing
from Plot.styles import mp_seaborn_light


def fitness(x):
    """
    示例目标函数 z = sin(x)+cos(y)
    :param x: 坐标 (x,y)
    :return: z
    """
    return np.sin(x[0]) + np.cos(x[1])


# 最初解、上下界
init_sol = [0.5, 0.5]
ub = [5, 5]
lb = [-5, -5]

# 退火算法开始
model = SimulatedAnnealing()
model.fit(fitness, init_sol, ub=ub, lb=lb, n_iters=300, init_temperature=100)

best_sol = model.best_sol
best_cost = model.best_costs[-1]
print("最优解: x =", best_sol, ", f(x) =", best_cost)

# 绘制原始函数图像和最优点
plt.style.use(mp_seaborn_light())

model.plot_iter_history()
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.sin(X1) + np.cos(X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.scatter(model.best_sol[0], model.best_sol[1], best_cost, color='red', label='最优解', s=100)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('目标函数值')
ax.set_title('退火算法求最优解')
ax.legend()
plt.show()
