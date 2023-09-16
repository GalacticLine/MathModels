""" 使用sklearn进行降维 """
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Plot.styles import mp_seaborn_light

# 加载葡萄酒数据集
wine = load_wine()
data = wine.data

# 标准化
data = StandardScaler().fit_transform(data)

# 计算不同成分数下的累计功率率
x = range(0, data.shape[1])
evr_s = []
for n in x:
    pca = PCA(n)
    pca.fit(data)
    cum_evar = np.sum(pca.explained_variance_ratio_)
    evr_s.append(cum_evar)

# 绘制图像
plt.style.use(mp_seaborn_light())

plt.figure()
plt.plot(x, evr_s)
plt.title('累计贡献率随成分数变化曲线')
plt.xlabel('成分数')
plt.ylabel('累计贡献率')

# pca 降维
pca = PCA(2)
tran = pca.fit_transform(data)

plt.figure()
plt.scatter(tran[:, 0], tran[:, 1], c=wine.target,s=50)
plt.title('双主成分散点图')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.show()
