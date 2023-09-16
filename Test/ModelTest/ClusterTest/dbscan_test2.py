"""
使用sklearn中的DBSCAN算法进行聚类分析
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from Plot.styles import mp_seaborn_light

# 生成散点数据
np.random.seed(0)
data = datasets.make_circles(4000, factor=0.5, noise=.05)[0]

# 聚类分析
model = DBSCAN(eps=0.05, min_samples=10)
model.fit(data)
print('聚类数：', np.sum(np.unique(model.labels_) != -1))

# 绘制图像
plt.style.use(mp_seaborn_light())
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=model.labels_, s=50)
plt.xlabel('x')
plt.ylabel('y')
plt.title('DBSCAN 聚类散点图')
plt.show()
