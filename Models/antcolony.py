""" 蚁群算法 """
import numpy as np


class AntColony:

    def __init__(self, n_ants, n_iters, pheromone_weight, inspire_weight, evaporate_rate):
        """
        蚁群算法类，解决旅行商问题（TSP）

        :param n_ants: 蚂蚁数量
        :param n_iters: 迭代次数
        :param pheromone_weight: 信息素权重
        :param inspire_weight: 启发式因子权重
        :param evaporate_rate: 信息素蒸发率
        """
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.pheromone_weight = pheromone_weight
        self.inspire_weight = inspire_weight
        self.evaporate_rate = evaporate_rate

    def optimize(self, dists):
        """
        使用蚁群算法优化路径。
        :param dists:  各个城市之间的距离
        :return: 最优路径，最短路径的距离
        """

        # 初始化信息素
        dists = np.asarray(dists)
        n = dists.shape[0]
        pheromones = np.ones((n, n))

        best = None
        shortest_dist = np.inf

        # 开始迭代
        for _ in range(self.n_iters):
            paths = []

            # 每只蚂蚁的路径构建过程
            for _ in range(self.n_ants):
                path = self.construct_path(pheromones, dists)
                dist = np.sum(dists[path[:-1], path[1:]])
                paths.append(path)

                # 更新最优解
                if dist < shortest_dist:
                    shortest_dist = dist
                    best = path

            # 信息素蒸发
            pheromones *= (1 - self.evaporate_rate)

            # 更新信息素
            for path in paths:
                path_dist = np.sum(dists[path[:-1], path[1:]])
                pheromones[path[:-1], path[1:]] += 1 / path_dist
        return best, shortest_dist

    def construct_path(self, pheromones, dists, r_tol=1e-10):
        """
        构建路径。
        :param pheromones: 信息素矩阵
        :param dists: 距离矩阵，表示各个城市之间的距离
        :param r_tol: 微小值
        :return: 路径
        """
        n = pheromones.shape[0]
        cities = np.arange(n)
        start = np.random.randint(n)
        unvisited = np.ones(n, dtype=bool)
        unvisited[start] = False
        path = [start]

        while np.any(unvisited):
            current = path[-1]

            # 计算当前城市与未访问的城市之间的信息素和启发式值
            pheromone = pheromones[current][unvisited]
            heuristic = 1 / (dists[current][unvisited] + r_tol)

            # 计算选择下一个城市的概率
            p = pheromone ** self.pheromone_weight
            p *= heuristic ** self.inspire_weight
            p /= np.sum(p)

            # 根据概率选择下一个城市
            next_city = np.random.choice(cities[unvisited], p=p)
            path.append(next_city)
            unvisited[next_city] = False

        return path
