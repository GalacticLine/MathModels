import numpy as np
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, ndim, bounds):
        """
        粒子群中单个粒子
        :param ndim: 粒子的维度数量
        :param bounds: 边界条件
        """
        position = []  # 位置
        velocity = []  # 速度
        lb, ub = bounds

        # 随机初始化粒子的位置和速度
        for _ in range(ndim):
            position.append(np.random.uniform(lb, ub))
            velocity.append(np.random.uniform(-1, 1))

        self.position = position  # 当前位置
        self.velocity = velocity  # 当前速度
        self.best_position = position  # 粒子历史最佳位置
        self.best_fitness = np.inf  # 粒子历史最佳适应度

    def update_position(self, ndim, bounds):
        """
        粒子更新位置
        :param ndim: 粒子的维度数量
        :param bounds: 边界条件
        :return:
        """
        position = self.position
        lb, ub = bounds
        # 更新粒子的位置
        for i in range(ndim):
            position[i] += self.velocity[i]

            # 粒子位置超出边界时进行修正
            if position[i] < lb:
                position[i] = lb
            elif position[i] > ub:
                position[i] = ub

        self.position = position

    def update_velocity(self, ndim, best_position):
        """
        粒子更新速度
        :param ndim: 粒子的维度数量
        :param best_position: 全局最佳位置
        :return:
        """
        w = 0.5  # 惯性权重
        c = 1  # 自我认知因子
        s = 1  # 社会认知因子

        # 更新粒子的速度
        for i in range(ndim):
            deta1 = self.best_position[i] - self.position[i]
            deta2 = best_position[i] - self.position[i]

            # 自我认知和社会认知
            cognitive = c * np.random.random() * deta1
            social = s * np.random.random() * deta2

            self.velocity[i] = w * self.velocity[i] + cognitive + social


class ParticleSwarm:
    def __init__(self, num, ndim, bounds=(-1, 1), max_iters=100, seed=42):
        """
        粒子群算法 解决优化问题 (最小化)

        :param num: 粒子数量
        :param ndim: 维度数量
        :param bounds: 边界条件
        :param max_iters: 最大迭代次数
        :param seed: 随机种子
        """
        np.random.seed(seed)

        self.num = num
        self.ndim = ndim
        self.bounds = bounds
        self.max_iters = max_iters
        self.fitness_history = []

    def fit(self, func):
        """
        使用粒子群优化算法进行优化

        :param func: 适应度函数，此函数输入粒子位置列表，输出适应度
        :return: 最优位置，最优适应度
        """
        n = self.ndim
        bounds = self.bounds

        particles = [Particle(n, bounds) for _ in range(self.num)]  # 创建粒子群
        best_position = None  # 全局最优位置
        best_fitness = np.inf  # 全局最优适应度

        # 粒子群优化迭代
        for _ in range(self.max_iters):
            for particle in particles:
                # 计算当前位置的适应度
                fitness = func(particle.position)

                # 更新粒子的最优位置、最佳适应度
                if fitness < particle.best_fitness:
                    particle.best_position = particle.position
                    particle.best_fitness = fitness

                # 更新全局最优位置、最佳适应度
                if fitness < best_fitness:
                    best_position = particle.position
                    best_fitness = fitness

                particle.update_velocity(n, best_position)  # 更新粒子的速度
                particle.update_position(n, bounds)  # 更新粒子的位置

            self.fitness_history.append(best_fitness)
        return best_position, best_fitness

    def plot(self):
        """
        绘制适应度曲线图
        """
        plt.plot(np.arange(len(self.fitness_history)), self.fitness_history)
        plt.xlabel('迭代数')
        plt.ylabel('适应度')
        plt.title('适应度随迭代数变化曲线')
        plt.show()
