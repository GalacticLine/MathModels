""" 聚类模型 """
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from DataProcess.decomposer import PCA
from Plot.functions import plot_cluster


class ClusterBase:

    def __init__(self):
        """
        聚类模型基类
        """
        self.data = None
        self.labels = None

    def plot_cluster(self,
                     title='聚类散点图',
                     x_label='x',
                     y_label='y',
                     z_label='z'):
        """
        绘制聚类散点图。

        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :param z_label: Z轴标题，当绘制三维图像时启用
        :return:
        """
        plot_cluster(self.data, self.labels, title, x_label, y_label, z_label)


class Kmeans(ClusterBase):

    def __init__(self):
        """
        K-means 聚类
        """
        super().__init__()
        self.centers = None
        self.sse = None
        self.n_iters = None

    @staticmethod
    def _fit(data, k, n_iters, random_seed=0, r_tol=1e-8):
        """
        k-means聚类主函数
        :param data: 数据
        :param k: 聚类数
        :param n_iters: 最大迭代次数
        :param random_seed: 随机种子
        :param r_tol: 微小值
        :return: 标签，中心，SSE（误差平方和）
        """
        np.random.seed(random_seed)

        # 从数据中随机选择k个样本作为初始中心点
        centers_idx = np.random.choice(range(len(data)), size=k, replace=False)
        centers = data[centers_idx]

        labels = None
        dists = None

        # 开始迭代
        for _ in range(n_iters):

            # 计算每个样本点到所有中心点的距离
            dists = np.sum((data - centers[:, np.newaxis]) ** 2, axis=-1)

            # 为每个样本指定最近中心点
            labels = np.argmin(dists, axis=0)

            new_centers = np.empty_like(centers)
            for i in range(k):
                # 如果存在指定类别的样本，将其均值作为新中心点，否则，中心点不变
                if np.any(labels == i):
                    new_centers[i] = np.mean(data[labels == i], axis=0)
                else:
                    new_centers[i] = centers[i]

            # 如果当前中心点与新中心点非常接近，则停止迭代
            if np.allclose(centers, new_centers, rtol=r_tol):
                break

            # 更新中心点
            centers = new_centers

        # 计算误差平方和
        sse = np.sum(np.min(dists, axis=0))

        return centers, labels, sse

    def fit(self, data, k, n_iters=100, random_seed=0, r_tol=1e-8):
        """
        k-means聚类主函数
        :param data: 数据
        :param k: 聚类数
        :param n_iters: 最大迭代次数
        :param random_seed: 随机种子
        :param r_tol: 微小值
        :return: 标签，中心
        """
        centers, labels, sse = self._fit(data, k, n_iters, random_seed, r_tol)

        self.data = data
        self.labels = labels
        self.centers = centers
        self.sse = sse
        self.n_iters = n_iters

    def plot_pca(self,
                 title='K-Means 双主成分散点图',
                 x_label='主成分1',
                 y_label='主成分2'):
        """
        绘制K聚类散点图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题

        :return:
        """
        data = self.data
        labels = self.labels
        centers = self.centers

        pca = PCA(2)
        data_pca = pca.fit(data)
        centers_pca = pca.fit(centers)

        plt.figure()
        for label in range(len(centers)):
            mask = labels == label
            plt.scatter(data_pca[mask, 0], data_pca[mask, 1], s=50)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=150, c='r')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    def plot_sse(self,
                 k,
                 title='K-Means SSE图',
                 x_label='聚类数',
                 y_label='误差平方和(SSE)'):
        """
        绘制SSE图
        :param k: 聚类数
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :return:
        """
        xs = range(1, k + 1)
        sse = [self._fit(self.data, x, self.n_iters)[-1] for x in xs]

        plt.figure()
        plt.plot(xs, sse, 'o-', markersize=8, markerfacecolor='w')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(['SSE'])

    def model_test(self):
        """
        聚类模型检验，包括轮廓系数、DBI系数、CH系数。
        """
        labs = self.labels
        data = self.data
        df = pd.DataFrame({
            '轮廓系数': silhouette_coefficient(data, labs),
            'DBI 系数': dbi_coefficient(data, labs),
            'CH 系数': ch_coefficient(data, labs)
        }, index=[0])
        return df


class DBSCAN(ClusterBase):

    def __init__(self):
        """
        DBSCAN 聚类
        """
        super().__init__()

    def fit(self, data, eps, min_samples):
        """
        DBSCAN聚类主函数。

        :param data: 数据
        :param eps: 邻域半径
        :param min_samples: 最小邻域样本数
        :return:
        """

        n = data.shape[0]
        labels = np.zeros(n, dtype=int)
        cluster_id = 1

        for i in range(n):
            # 如果邻域样本已经分类
            if labels[i] != 0:
                continue

            # 获取邻域样本
            neighbors = np.where(np.linalg.norm(data[i] - data, axis=1) <= eps)[0]
            neighbors = neighbors[neighbors != i]
            
            if len(neighbors) >= min_samples:
                labels[i] = cluster_id

                while len(neighbors) > 0:
                    neighbor = neighbors[0]

                    # 如果邻域样本未分类
                    if labels[neighbor] == 0:
                        labels[neighbor] = cluster_id

                        # 获取邻域样本的领域样本
                        neighbor_neighbors = np.where(np.linalg.norm(data[neighbor] - data, axis=1) <= eps)[0]
                        neighbor_neighbors = neighbor_neighbors[neighbor_neighbors != neighbor]
                        if len(neighbor_neighbors) >= min_samples:
                            neighbors = np.concatenate((neighbors, neighbor_neighbors))

                    neighbors = neighbors[1:]

                cluster_id += 1
            else:
                # 噪声点
                labels[i] = -1
        # 储存结果
        self.data = data
        self.labels = labels

    @property
    def n_cluster(self):
        return np.sum(np.unique(self.labels) != -1)


class HierarchicalCluster(ClusterBase):

    def __init__(self):
        """
        层次聚类
        """
        super().__init__()
        self.dists = None

    def fit(self, data, k):
        """
        层次聚类主函数
        :param data: 数据
        :param k: 聚类数
        :return:
        """
        # 计算数据样本之间的欧式距离
        dists = np.linalg.norm(data[:, np.newaxis] - data, axis=2)

        # 初始化标签
        labels = np.arange(data.shape[0])

        while len(np.unique(labels)) > k:
            # 找到当前距离矩阵中最小距离的索引
            tri = np.triu_indices(dists.shape[0], k=1)
            min_idx = np.argmin(dists[tri])
            i = tri[0][min_idx]
            j = tri[1][min_idx]

            # 标签替换
            labels[labels == j] = i

            # 更新标签
            labels[labels > j] -= 1

            # 删除距离矩阵中第j行和第j列
            dists = np.delete(dists, j, axis=0)
            dists = np.delete(dists, j, axis=1)

            # 使用最小距离更新距离矩阵
            dists[i, :] = np.minimum(dists[i, :], dists[:, i])
            dists[:, i] = dists[i, :]

        # 储存结果
        self.data = data
        self.labels = labels
        self.dists = dists

    def plot_dendrogram(self,
                        title='层次聚类树状图',
                        x_label='样本索引',
                        y_label='距离'):
        """
        绘制层次聚类树状图
        :param title: 标题
        :param x_label: x轴标题
        :param y_label: y轴标题
        :return:
        """
        linkage_matrix = linkage(self.data, method='ward')
        plt.figure()
        dendrogram(linkage_matrix)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


def silhouette_coefficient(data, labels):
    """
    计算轮廓系数
    :param data: N,M 特征序列
    :param labels: 聚类结果标签
    :return: 轮廓系数
    """
    n = len(data)
    k = len(np.unique(labels))
    silhouette = np.zeros(n)

    # 循环计算轮廓系数
    for i in range(n):
        cluster_data = data[labels == labels[i]]
        a = np.mean(np.linalg.norm(cluster_data - data[i], axis=1))
        b = np.inf
        for j in range(k):
            if j != labels[i]:
                other_cluster_data = data[labels == j]
                b = min(b, np.mean(np.linalg.norm(other_cluster_data - data[i], axis=1)))
        silhouette[i] = (b - a) / max(a, b)

    return np.mean(silhouette)


def dbi_coefficient(data, labels):
    """
    计算DBI系数
    :param data: N,M 特征序列
    :param labels: 聚类结果标签
    :return: DBI系数
    """
    k = len(np.unique(labels))
    centroids = np.zeros((k, data.shape[1]))
    dbi_vals = np.zeros(k)

    # 计算聚类中心
    for i in range(k):
        cluster_data = data[labels == i]
        centroids[i] = np.mean(cluster_data, axis=0)

    # 计算每个聚类簇的DBI值
    for i in range(k):
        cluster_data_i = data[labels == i]
        a = np.mean(np.linalg.norm(cluster_data_i - centroids[i], axis=1))
        max_db = 0
        for j in range(k):
            if j != i:
                cluster_data_j = data[labels == j]
                b = np.mean(np.linalg.norm(cluster_data_j - centroids[j], axis=1))
                db = (a + b) / np.linalg.norm(centroids[i] - centroids[j])
                max_db = max(max_db, db)

        dbi_vals[i] = max_db

    return np.mean(dbi_vals)


def ch_coefficient(data, labels):
    """
    计算CH系数。
    :param data: N,M 特征序列
    :param labels: 聚类结果标签
    :return: CH系数
    """

    # 聚类数
    uni_label = np.unique(labels)
    centroids = np.array([np.mean(data[labels == label], axis=0) for label in uni_label])

    # 聚类簇与数据集的距离的平方
    data_mean = np.mean(data, axis=0)
    dists = np.sum((centroids - data_mean) ** 2, axis=1)

    # 计算组内离差平方和(WSS), 组间离差平方和(BSS)
    wss = np.sum([np.sum((data[labels == label] - centroids[i]) ** 2) for i, label in enumerate(uni_label)])
    bss = np.sum([np.sum(labels == label) * dists[i] for i, label in enumerate(uni_label)])

    # 计算CH系数
    k = len(uni_label)
    ch = bss / ((k - 1) * wss / (len(data) - k))
    return ch
