"""
kmeans聚类算法测试
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from Models.cluster import Kmeans
from Plot.styles import mp_seaborn_light

# 生成散点数据
np.random.seed(0)
centers = [[1.5, 1.5, 1.5], [0, 0, 0], [-1.5, -1.5, -1.5]]
data = make_blobs(n_samples=300, centers=centers, cluster_std=0.3)[0]

# 聚类分析
model = Kmeans()
model.fit(data, 3)
print("聚类标签：", model.labels)
print("聚类中心点：", model.centers)
print('模型检验:')
print(model.model_test())

# 绘制相关图像
plt.style.use(mp_seaborn_light())
model.plot_sse(10)
model.plot_pca()
model.plot_cluster('Kmeans 聚类散点图')
centers = np.asarray(centers)
plt.gca().scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r', s=100)
plt.show()
