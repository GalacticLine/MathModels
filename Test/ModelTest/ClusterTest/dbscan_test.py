"""
DBSCAN算法测试
"""
import numpy as np
from sklearn import datasets
from Models.cluster import DBSCAN
import matplotlib.pyplot as plt
from Plot.styles import mp_seaborn_light

# 生成散点数据
np.random.seed(0)
data = datasets.make_circles(4000, factor=0.5, noise=.05)[0]

# 聚类分析
model = DBSCAN()
model.fit(data, 0.05, 10)
print('聚类数：', model.n_cluster)

# 绘制图像
plt.style.use(mp_seaborn_light())
model.plot_cluster('DBSCAN 聚类散点图')
plt.show()
