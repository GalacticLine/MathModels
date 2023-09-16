""" 元胞自动机 """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors


class GameOfLife:

    def __init__(self, grid_size=2, space_size=100, live_rule=(2, 3), die_rule=(3,)):
        """
        生命游戏
        :param grid_size: 元胞周围网格大小，默认2，即以元胞为中心3x3的范围
        :param space_size: 模拟空间大小
        :param live_rule: 存活规则，默认(2,3)
        :param die_rule:  死亡规则，默认(3,)
        """
        self.space_size = space_size
        self.space = np.random.choice([0, 1], (space_size, space_size))  # 0死亡 1存活

        self.live_rule = live_rule
        self.dead_rule = die_rule
        self.grid_size = grid_size

        self.img = None

    def update(self, frame):
        """
        更新函数，用于计算下一个时刻的细胞状态
        :param frame:
        :return:
        """
        space = self.space
        space_size = self.space_size
        grid_size = self.grid_size

        new_grid = np.zeros_like(space)
        for i in range(space_size):
            for j in range(space_size):
                # 计算周围细胞的活跃数量
                neighbors_sum = np.sum(space[max(0, i - 1):i + grid_size, max(0, j - 1):j + grid_size]) - space[i, j]

                # 根据生命游戏的规则更新细胞状态
                if space[i, j] == 1:
                    if neighbors_sum in self.live_rule:
                        new_grid[i, j] = 1
                else:
                    if neighbors_sum in self.dead_rule:
                        new_grid[i, j] = 1

        self.space = new_grid
        self.img.set_array(new_grid)
        return self.img,

    def start(self, title='生命游戏'):
        """
        生命游戏开始
        :param title: 标题
        :return:
        """
        fig, ax = plt.subplots()
        self.img = ax.imshow(self.space, cmap='binary')
        ani = animation.FuncAnimation(fig, self.update, frames=100, interval=200, blit=True)
        plt.title(title)
        plt.show()


class TreeFire:
    def __init__(self, grid_size=2, space_size=100, density=0.9, fire_prob=0.5, tree_regrow=0.01):
        """
        森林火灾模拟
        :param grid_size: 元胞周围网格大小，默认2，即以元胞为中心3x3的范围
        :param space_size: 模拟空间大小
        :param density: 初始生成树木的概率
        :param fire_prob: 火灾扩散到附近树木的概率
        :param tree_regrow: 空地新生树木的概率
        """
        # 随机一处火灾发生地
        space = np.random.choice([0, 1], (space_size, space_size), p=[1 - density, density])
        indices = np.where(space == 1)
        random_index = np.random.choice(indices[0].size)
        space[indices[0][random_index], indices[1][random_index]] = 2

        self.space = space
        self.space_size = space_size
        self.grid_size = grid_size
        self.fire_p = fire_prob
        self.tree_regrow = tree_regrow
        self.img = None

    def update(self, frame):
        """
        更新函数，用于计算下一个时刻的细胞状态
        :param frame:
        :return:
        """
        space = self.space
        space_size = self.space_size
        grid_size = self.grid_size

        new_grid = np.zeros_like(space)
        for i in range(space_size):
            for j in range(space_size):
                if space[i, j] == 2:
                    # 烧成空地
                    new_grid[i, j] = 0
                elif space[i, j] == 1:
                    num_burning = np.sum(space[max(0, i - 1):i + grid_size, max(0, j - 1):j + grid_size] == 2)
                    if num_burning > 0 and np.random.rand() < self.fire_p:
                        # 点燃树木
                        new_grid[i, j] = 2
                    else:
                        new_grid[i, j] = 1
                elif space[i, j] == 0:
                    if np.random.rand() < self.tree_regrow:
                        # 新生树木
                        new_grid[i, j] = 1

        self.space = new_grid
        self.img.set_array(new_grid)
        return self.img,

    def start(self, title='森林火灾模拟'):
        """
        火灾模拟开始
        :param title: 绘图标题
        :return:
        """
        fig, ax = plt.subplots()
        cmap = colors.ListedColormap(['#FFE4B5', 'g', 'r'])  # 修改颜色表，增加了一种颜色

        bounds = [-0.5, 0.5, 1.5, 2.5]  # 添加一些边界来区分不同的值
        norm = colors.BoundaryNorm(bounds, cmap.N)

        self.img = ax.imshow(self.space, cmap=cmap, norm=norm)
        ani = animation.FuncAnimation(fig, self.update, frames=100, interval=200, blit=True)
        plt.title(title)
        plt.show()




