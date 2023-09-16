"""
使用sklearn中的kmeans聚类算法实现进行聚类分析
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from Plot.styles import mp_seaborn_light

# 生成散点数据
np.random.seed(0)
centers = [[1.5, 1.5, 1.5], [0, 0, 0], [-1.5, -1.5, -1.5]]
data = make_blobs(n_samples=300, centers=centers, cluster_std=0.3)[0]

# 聚类分析
kmeans = KMeans(n_clusters=3, random_state=0, n_init=1)
kmeans.fit(data)

labels = kmeans.labels_
silhouette = silhouette_score(data, labels)
dbi = davies_bouldin_score(data, labels)
ch = calinski_harabasz_score(data, labels)

print('聚类标签:', labels)
print('聚类中心', kmeans.cluster_centers_)
print("轮廓系数:", silhouette)
print("DBI系数:", dbi)
print('CH系数:', ch)

# 绘制图像
plt.style.use(mp_seaborn_light())
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in np.unique(labels):
    mask = labels == i
    ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2], label=f'聚类 {i}', s=50)

centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r', s=100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Kmeans 聚类散点图')
ax.legend()
plt.show()
