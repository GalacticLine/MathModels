"""
使用sklearn中的DBSCAN算法实现进行聚类分析
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from Plot.styles import mp_seaborn_light

# 生成散点数据
np.random.seed(0)
data = make_blobs(n_samples=[15, 15, 15], centers=[[1, 1], [-2, 0], [0, -2]], cluster_std=0.4)[0]

# 聚类分析
model = AgglomerativeClustering(n_clusters=3)
model.fit(data)
print(model.labels_)

# 绘制图像
plt.style.use(mp_seaborn_light())
plt.figure()
linkage_matrix = linkage(data, method='ward')
dendrogram(linkage_matrix)
plt.title('层次聚类图')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.show()
