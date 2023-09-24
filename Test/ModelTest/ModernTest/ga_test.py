""" 遗传算法测试 """
from Models.genetic_algorithm import GeneticAlgorithm


def fitness_function(x):
    """ 适应度函数 """
    return x ** 2


ga = GeneticAlgorithm(fitness_function,
                      bounds=(-10, 10),
                      ch_length=16,
                      population_size=50,
                      mute_rate=0.01,
                      n_iters=100)

best_sol, best_fitness = ga.fit()
print("最优解: x =", best_sol, ", f(x) =", best_fitness)  # 打印最优解


def fitness_function(x):
    """ 适应度函数，求最小化问题时，添加负号 """
    return - x ** 2


ga = GeneticAlgorithm(fitness_function,
                      bounds=(-10, 10),
                      ch_length=16,
                      population_size=50,
                      mute_rate=0.01,
                      n_iters=100)

best_sol, best_fitness = ga.fit()
print("最优解: x =", best_sol, ", f(x) =", best_fitness)  # 打印最优解
