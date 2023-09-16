""" 遗传算法 """
import numpy as np


class GeneticAlgorithm:
    def __init__(self, func, bounds=(-1, 1), ch_length=8, mute_rate=0.01, population_size=50, n_iters=200):
        """
        遗传算法 解决优化问题 (最大化问题)
        :param func: 适应度函数
        :param bounds: 边界条件
        :param ch_length: 染色体长度
        :param population_size: 种群数量
        :param mute_rate: 变异率
        :param n_iters: 最大迭代次数
        """
        self.fun = func
        self.bounds = bounds
        self.ch_length = ch_length
        self.mute_rate = mute_rate
        self.population_size = population_size
        self.n_iters = n_iters

    def fit(self):
        """
        遗传算法主函数
        :return:
        """
        population_size = self.population_size
        population = np.random.randint(2, size=(population_size, self.ch_length)).tolist()

        best_fitness = -np.inf
        best_sol = None

        for generation in range(self.n_iters):
            # 选择优秀个体
            population = self.select(population)

            # 产生下一代种群
            next_population = []
            while len(next_population) < population_size:
                parent1, parent2 = [population[i] for i in np.random.choice(len(population), 2)]
                child1, child2 = self.cross(parent1, parent2)
                mutated_child1 = self.mute(child1)
                mutated_child2 = self.mute(child2)
                next_population.extend([mutated_child1, mutated_child2])  # 将变异后的子代加入下一代种群
            population = next_population[:population_size]

            # 更新最优解
            values = self.cal_fitness_values(population)
            best_index = np.argmax(values)
            max_fitness = values[best_index]

            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_sol = population[best_index]
        best_sol = self.decoding(best_sol)

        return best_sol, best_fitness

    def cal_fitness_values(self, population):
        """
        计算新种群中每个个体的适应度值
        :param population: 当前种群
        :return: 适应度集合
        """
        values = np.zeros(len(population))
        for i, chromosome in enumerate(population):
            fitness = self.fun(self.decoding(chromosome))
            values[i] = fitness
        return values

    def decoding(self, chromosome):
        """
        解码染色体
        :param chromosome: 染色体 (指定长度0,1集合)
        :return:
        """
        left, right = self.bounds

        # 将二进制转换为十进制
        binary = 0
        for bit in chromosome:
            binary = (binary << 1) | bit

        # 计算变量值
        value = left + binary / (2 ** self.ch_length - 1) * (right - left)
        return value

    def select(self, population):
        """
        选择过程
        :param population: 当前种群
        :return: 选择后的种群
        """
        # 计算每个个体的选择概率
        values = self.cal_fitness_values(population)
        probs = values / np.sum(values)

        # 随机选择指定数量的个体
        selector = np.random.choice(len(population), self.population_size, p=probs)
        selected = [population[i] for i in selector]
        return selected

    def cross(self, parent1, parent2):
        """
        交叉过程
        :param parent1: 父代染色体1
        :param parent2: 父代染色体2
        :return: 生成的子代染色体
        """
        # 随机选择交叉点，生成子代
        point = np.random.randint(1, self.ch_length - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mute(self, chromosome):
        """
        变异过程
        :param chromosome: 染色体
        :return: 变异后的染色体
        """
        mutated = []
        for bit in chromosome:
            # 是否进行变异
            if np.random.random() < self.mute_rate:
                bit = not bit
            mutated.append(bit)
        return mutated
