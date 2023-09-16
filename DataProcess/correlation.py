""" 相关性分析 """
import numpy as np
import pandas as pd
from scipy import stats
from Plot.functions import plot_matrix
from Tools.numpyHelper import np_rank


class Correlation:
    def __init__(self):
        """
        相关性分析辅助类。
        """
        self.func_name = None
        self.corr = None
        self.p_value = None
        self.color = None

    def fit(self, data, func_name):
        """
        相关性分析主函数。
        :param data: N,M 特征序列
        :param func_name: 函数名
        :return:
        """
        if func_name not in ['Spearman', 'Pearson', 'Kendall']:
            raise ValueError("请输入有效的函数名：'Spearman', 'Pearson', 或 'Kendall'")

        # 定义不同相关性计算函数和对应绘图颜色
        maps = {
            'Spearman': (stats.spearmanr, 'Greens'),
            'Pearson': (stats.pearsonr, 'Reds'),
            'Kendall': (stats.kendalltau, 'Blues')
        }

        func, color = maps[func_name]

        arr = np.asarray(data)
        n = arr.shape[1]

        # 初始化相关性矩阵和p值矩阵
        corr_s = np.zeros((n, n))
        p_values = np.zeros((n, n))

        # 计算相关性和p值
        for i in range(n):
            for j in range(i, n):
                corr, p_value = func(arr[:, i], arr[:, j])
                corr_s[i, j] = corr
                corr_s[j, i] = corr
                p_values[i, j] = p_value
                p_values[j, i] = p_value

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        df_corr = pd.DataFrame(corr_s, columns=data.columns, index=data.columns)
        df_p_value = pd.DataFrame(p_values, columns=data.columns, index=data.columns)

        # 存储结果
        self.color = color
        self.func_name = func_name
        self.corr = df_corr
        self.p_value = df_p_value

    def plot_corr(self):
        """
        绘制相关系数矩阵图。
        :return:
        """
        plot_matrix(self.corr, f"{self.func_name}相关性分析", cmap=self.color)


def spearman_corr(data):
    """
    计算Spearman相关性系数。
    :param data: N,M 特征序列
    :return: 相关系数矩阵
    """
    # 计算秩次矩阵
    ranks = np_rank(data)

    # 计算相关性系数矩阵
    n = data.shape[0]
    dist = np.linalg.norm(ranks[:, :, np.newaxis] - ranks[:, np.newaxis, :], axis=0)**2
    corr = 1 - 6 * dist / (n * (n ** 2 - 1))
    return corr


def pearson_corr(data):
    """
    计算Spearman相关性系数
    :param data: N,M 特征序列
    :return: 相关系数矩阵
    """
    return np.corrcoef(data, rowvar=False)


def kendall_corr(data):
    """
    计算kendall相关性系数。
    :param data: N,M 特征序列
    :return: 相关系数矩阵
    """
    n = data.shape[1]
    corr = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dat1 = data[:, i]
            dat2 = data[:, j]

            diff1 = dat1[:, np.newaxis] - dat1[np.newaxis, :]
            diff2 = dat2[:, np.newaxis] - dat2[np.newaxis, :]

            mul_diff = diff1 * diff2

            con = np.sum(mul_diff > 0)
            dis = np.sum(mul_diff < 0)

            tx = np.sum((diff1 == 0) & (diff2 != 0))
            ty = np.sum((diff1 != 0) & (diff2 == 0))

            cd = con + dis
            tau_b = (con - dis) / np.sqrt((cd + tx) * (cd + ty))

            corr[i][j] = tau_b
            corr[j][i] = tau_b

    return corr
