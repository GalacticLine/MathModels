""" 使用sklearn支持向量机模型进行鸢尾花数据集分类任务 """
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from Plot.styles import mp_seaborn_light

# 加载鸢尾花数据集
iris = datasets.load_iris()
x = iris.data[:, [0, 2]]
y = iris.target

# 将数据集分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建支持向量机模型
model = SVC(kernel='rbf', gamma='auto')

# 在训练集上训练模型
model.fit(x_train, y_train)

# 预测
y_pre = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pre)
print("准确率：", accuracy)

h = 0.01  # 网格步长
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 对网格点进行预测
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

# 绘制分类结果的图像
plt.style.use(mp_seaborn_light())
plt.pcolormesh(xx, yy, z, shading='gouraud')
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='w', s=100)
plt.xlabel('花萼长度')
plt.ylabel('花瓣长度')
plt.title('SVM 鸢尾花数据集分类')
plt.grid(False)
plt.show()
