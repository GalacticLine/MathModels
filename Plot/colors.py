""" 绘图颜色助手 """
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans


def color_to_hex(color):
    """
    颜色值转十六进制
    :param color: rgb颜色
    :return: 十六进制颜色
    """
    return '#{:02x}{:02x}{:02x}'.format(*color)


def color_to_lum(color):
    """
    颜色值转亮度值
    :param color: rgb颜色
    :return: 亮度
    """
    r, g, b = color
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


class ColorGetter:
    def __init__(self, arr):
        """
        取色器类
        :param arr: 矩阵
        """
        self.data = arr

    @classmethod
    def from_image(cls, path, num):
        """
        使用k-means聚类从图像获取颜色
        :param path: 图像路径
        :param num: 颜色数
        :return: 取色器实例
        """
        image = Image.open(path)
        image = image.resize((300, 300))
        image = np.array(image.im)
        kmeans = KMeans(n_clusters=num, n_init=10)
        kmeans.fit(image)
        colors = kmeans.cluster_centers_
        colors = colors.astype(int)
        obj = cls(colors)
        return obj

    def to_cmap(self):
        """
        颜色数组转颜色表
        :return: 颜色表
        """
        colors = sorted(self.data / 255, key=color_to_lum)
        colors.reverse()
        cmap = ListedColormap(colors)
        return cmap

    def plot(self):
        """
        绘制色卡
        :return:
        """
        plt.figure()
        plt.imshow([self.data], aspect='auto')
        plt.axis('off')
        for i, color in enumerate(self.data):
            if color_to_lum(color) < 0.5:
                text_color = 'white'
            else:
                text_color = 'black'

            plt.text(i, 0, color_to_hex(color), ha='center', va='bottom', color=text_color, rotation=90)
        plt.tight_layout()

    def get_hex_data(self, no_sharp=False):
        """
        获取十六进制颜色格式
        :param no_sharp: 是否不需要 '#'
        :return: 颜色代码
        """
        colors = np.apply_along_axis(color_to_hex, 1, self.data)
        if no_sharp:
            return [x.replace('#', '') for x in list(colors)]
        else:
            return colors
