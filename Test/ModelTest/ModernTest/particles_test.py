""" 粒子群算法测试 """
import numpy as np
from Models.particles import ParticleSwarm


def fitness_func(x):
    """ 适应度函数 """
    return np.sum(x)


# 粒子群算法
ps = ParticleSwarm(num=30,
                   ndim=2,
                   bounds=(-10, 10),
                   max_iters=100)

best_position, best_fitness = ps.fit(fitness_func)
print("最优解: x =", best_position, ", f(x) =", best_fitness)  # 打印最优解
