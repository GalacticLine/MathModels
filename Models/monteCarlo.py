""" 蒙特卡洛 """
import numpy as np
from matplotlib import pyplot as plt


class MonteCarlo:
    def __init__(self):
        """
        蒙特卡洛函数模拟类, 解决2维、3维图形模拟问题。
        """
        self.x = None
        self.y = None
        self.z = None
        self.approx = None
        self.mode = None

    def fit2d(self, x_min, x_max, y_min, y_max, num, masker):
        """
        模拟函数，用于计算函数的近似积分值。
        :return: 返回模拟散点的x坐标、y坐标和近似积分值。
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :param num:
        :param masker:
        :return:
        """

        x = np.random.uniform(x_min, x_max, num)
        y = np.random.uniform(y_min, y_max, num)

        inside = masker(x, y)
        approx = np.sum(inside > 0) / np.size(inside) * ((x_max - x_min) * (y_max - y_min))

        self.x = x[inside]
        self.y = y[inside]
        self.approx = approx
        self.mode = '2D'

    def fit3d(self, x_min, x_max, y_min, y_max, z_min, z_max, num, masker):
        """
        蒙特卡洛模拟函数，用于计算函数的近似积分值。
        :return: 返回模拟散点的x坐标、y坐标、z坐标和近似积分值。
        """

        x = np.random.uniform(x_min, x_max, num)
        y = np.random.uniform(y_min, y_max, num)
        z = np.random.uniform(z_min, z_max, num)

        inside = masker(x, y, z)
        approx = np.sum(inside > 0) / np.size(inside) * ((x_max - x_min) * (y_max - y_min) * (z_max - z_min))

        self.x = x[inside]
        self.y = y[inside]
        self.z = z[inside]
        self.approx = approx
        self.mode = '3D'

    def plot_mc(self, title="蒙特卡罗模拟图"):
        """
        绘制三维蒙特卡洛模拟的散点图。
        """
        if self.mode == '2D':
            plt.figure()
            plt.title(title)
            plt.scatter(self.x, self.y, s=50)

        if self.mode == '3D':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.x, self.y, self.z, s=50)
            ax.set_title(title)


