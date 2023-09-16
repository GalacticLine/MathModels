""" 数据降维 """
import numpy as np
from matplotlib import pyplot as plt


class PCABase:
    def __init__(self, n_components):
        """
        主成分分析基类。
        :param n_components: 输出维度数
        """
        self.n_components = n_components
        self.va = None
        self._evar = None
        self._cum_evar = None

    @property
    def evar(self):
        """
        计算方差贡献度。
        :return: 方差贡献度
        """
        if self._evar is None:
            va = self.va
            n = self.n_components

            indices = np.sort(np.argpartition(-va, n)[:n])
            evar = va[indices] / np.sum(va)

            self._evar = evar
        return self._evar

    @property
    def cum_evar(self):
        """
        计算累计方差贡献度。
        :return: 累计方差贡献度
        """
        if self._cum_evar is None:
            self._cum_evar = np.cumsum(self.evar)
        return self._cum_evar

    def plot_evar(self):
        """
        绘制方差贡献率图和累计方差贡献率阶梯图。
        :return:
        """
        n = self.n_components
        x = range(n)

        plt.figure(figsize=(14, 8))

        plt.subplot(121)
        plt.bar(x, self.evar)
        plt.xticks(x)
        plt.title('方差贡献度柱形图')
        plt.xlabel('主成分')
        plt.ylabel('方差贡献度')

        plt.subplot(122)
        plt.step(x, self.cum_evar, where='post')
        plt.xticks(x)
        plt.title('累计方差贡献度阶梯图')
        plt.xlabel('主成分')
        plt.ylabel('累计方差贡献度')

        plt.subplots_adjust(wspace=0.4)


class PCA(PCABase):
    def __init__(self, n_components):
        """
        主成分分析法 PCA

        基于线性变换的方法，通过保留最大方差的方式找到数据中的主要成分。
        :param n_components: 输出维度数
        """
        super().__init__(n_components)

    def fit(self, data):
        """
        主成分分析主函数。
        :param data: N,M 特征序列
        :return: 降维数据
        """
        n = self.n_components

        # 对数据进行中心化处理
        data = np.asarray(data)
        data = data - np.mean(data, axis=0)

        # 计算协方差矩阵并进行特征值分解
        cov = np.cov(data.T)
        va, vc = np.linalg.eigh(cov)

        # 从大到小排序并取出的前n个特征向量作为主成分
        sort_indices = np.argsort(va)[::-1]
        top = vc[:, sort_indices[:n]]

        # 通过线性变换将原数据映射到主成分空间
        tran = np.dot(data, top)

        # 储存结果
        self.n_components = n
        self.va = va
        self._evar = None
        self._cum_evar = None

        return tran


class KernelPCA(PCABase):

    def __init__(self, n_components):
        """
        核主成分分析
        :param n_components: 输出维度数
        """
        super().__init__(n_components)

    def fit(self, data):
        """
        kernelPCA 主函数，核函数类型为 rbf.
        :param data: N,M 特征序列
        :return: 降维数据
        """
        n_components = self.n_components

        # 对数据进行中心化处理
        data = np.asarray(data)
        data = data - np.mean(data, axis=0)

        # 计算欧几里得范数平方
        dists = np.linalg.norm(data[:, np.newaxis] - data, axis=2) ** 2

        # 计算高斯核矩阵
        gamma = 1 / data.shape[1]
        kernel = np.exp(-gamma * dists)

        # 计算中心化的核矩阵
        n = data.shape[0]
        unit = np.ones((n, n)) / n
        kernel_centered = kernel - unit @ kernel - kernel @ unit + unit @ kernel @ unit

        # 获取中心化核矩阵的特征值和特征向量
        va, vc = np.linalg.eigh(kernel_centered)

        # 根据输出维度数选择特征向量
        tran = vc[:, -n_components:][:, ::-1]

        # 储存结果
        self.n_components = n_components
        self.va = va
        self._evar = None
        self._cum_evar = None

        return tran
