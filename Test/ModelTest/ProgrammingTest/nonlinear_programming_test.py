""" 使用scipy进行非线性规划 """
import numpy as np
from scipy.optimize import minimize


def objective(x):
    """ 目标函数 """
    return x[0] ** (5 / 2) - np.exp(x[1] - 1) ** 2


def ineq1(x):
    """ 不等式约束条件1 """
    return (x[0] - 1) ** 2 + 3 * x[1] + 3


def ineq2(x):
    """ 不等式约束条件2 """
    return -x[0] ** 3 - x[1] + 10


def ineq3(x):
    """ 不等式约束条件3 """
    return x[0] ** 3 - x[1] - 10


def eq1(x):
    """ 等式约束条件1 """
    return x[0] + x[1] - 1


# 约束条件集合
cons = (
    {'type': 'ineq', 'fun': ineq1},
    {'type': 'ineq', 'fun': ineq2},
    {'type': 'ineq', 'fun': ineq3},
    {'type': 'eq', 'fun': eq1}
)

# 初始解
x0 = np.array([1, 0])

# 边界条件
bounds = ([0, None], [-10, None])

# 非线性规划求解
res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

print("目标的最优解:", res.x)
print("目标的最小值:", res.fun)
print("优化是否成功:", res.success)
print("结束时的消息:", res.message)
