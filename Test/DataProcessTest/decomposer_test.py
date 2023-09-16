""" 主成分分析降维 """
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from DataProcess.decomposer import PCA
from Plot.styles import mp_seaborn_light

# 加载葡萄酒数据集
wine = load_wine()
data = wine.data

# 标准化
data = StandardScaler().fit_transform(data)

model = PCA(6)
tran = model.fit(data)
print(tran)
print('各主成分贡献度:', model.evar)
print('最大累计贡献度:', model.cum_evar[-1])

plt.style.use(mp_seaborn_light())
model.plot_evar()
plt.show()
