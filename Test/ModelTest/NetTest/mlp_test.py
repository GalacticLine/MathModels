"""
使用是sklearn库中用于实现多层感知器MLP类进行函数拟合，
多层感知器是一种基于反向传播算法的BP神经网络
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from Plot.functions import plot_network
from Plot.styles import mp_seaborn_light


def function(x):
    return x ** 2 + 2 * x + 1


# 生成训练数据
x_train = np.linspace(-10, 10, 100).reshape(-1, 1)
y_train = function(x_train)

# 创建并训练多层感知器
model = MLPRegressor(hidden_layer_sizes=4,
                     activation='relu',
                     random_state=42,
                     learning_rate_init=1,
                     max_iter=100)
model.fit(x_train, y_train)

# 使用训练好的模型进行预测
x_test = np.linspace(-10, 10, 1000).reshape(-1, 1)
y_pre = model.predict(x_test)

# 绘制数据和拟合结果
plt.style.use(mp_seaborn_light())
plot_network(1, [4], 1)
plt.figure()
plt.scatter(x_train, y_train, label='原数据', s=50)
plt.plot(x_test, y_pre, 'b', label='拟合曲线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('多层感知器拟合图')
plt.legend()
plt.show()
