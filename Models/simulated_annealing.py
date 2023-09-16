""" 退火算法 """
import numpy as np
from matplotlib import pyplot as plt


class SimulatedAnnealing:

    def __init__(self):
        self.best_sol = None
        self.best_costs = None
        self.iter_history = None

    def fit(self, func, init_sol, ub=None, lb=None, init_temperature=100.0, cooling_rate=0.01, n_iters=1000,seed=0):
        """
        退火算法 解决优化问题 (最小化)

        :param func: 目标函数
        :param init_sol: 初始解
        :param ub: 上限
        :param lb: 下限
        :param init_temperature: 初始温度
        :param cooling_rate: 降温速率
        :param n_iters: 迭代次数
        :param seed: 随机种子
        :return:
        """
        np.random.seed(seed)

        # 当前解
        current_sol = np.asarray(init_sol)

        # 最优解，最优成本
        best_sol = current_sol
        best_costs = [func(best_sol)]

        iter_history = [0]

        # 开始循环
        for i in range(n_iters):
            # 生成新解
            new_sol = current_sol + np.random.uniform(-1, 1, size=current_sol.shape)

            # 将解限制在上下限内
            if ub is not None and lb is not None:
                new_sol = np.clip(new_sol, lb, ub)

            # 计算当前温度，根据接受概率判断是否接受新解
            temperature = init_temperature * np.exp(-cooling_rate * i)
            p = self.accept_probability(func(current_sol), func(new_sol), temperature)
            if p > np.random.rand():
                current_sol = new_sol

            # 更新最佳解
            best_cost = func(best_sol)
            if func(current_sol) < best_cost:
                best_sol = current_sol

            best_costs.append(best_cost)
            iter_history.append(i + 1)

        # 储存结果
        self.best_sol = best_sol
        self.best_costs = best_costs
        self.iter_history = iter_history

    @staticmethod
    def accept_probability(old_cost, new_cost, temperature):
        """
        计算接受概率
        :param old_cost: 当前成本
        :param new_cost: 新解成本
        :param temperature: 温度
        :return: 接受概率
        """
        if new_cost < old_cost:
            return 1.0
        else:
            return np.exp((old_cost - new_cost) / temperature)

    def plot_iter_history(self):
        """
        绘制目标函数值随迭代次数的变化曲线
        :return:
        """
        plt.figure()
        plt.plot(self.iter_history, self.best_costs)
        plt.xlabel('迭代数')
        plt.ylabel('目标函数值')
        plt.title('目标函数值随迭代次数变化曲线')
