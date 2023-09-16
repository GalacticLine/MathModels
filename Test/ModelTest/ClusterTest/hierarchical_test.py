"""
层次聚类算法测试
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from Models.cluster import HierarchicalCluster
from Plot.styles import mp_seaborn_light

# 生成散点数据
np.random.seed(0)
data = make_blobs(n_samples=[15, 15, 15], centers=[[1, 1], [-2, 0], [0, -2]], cluster_std=0.4)[0]

# 聚类分析
model = HierarchicalCluster()
model.fit(data, 3)
print(model.labels)

# 绘制图像
plt.style.use(mp_seaborn_light())
model.plot_cluster('层次聚类散点图')
model.plot_dendrogram()
plt.show()

