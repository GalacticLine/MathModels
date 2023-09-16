""" bp神经网络测试 实现简单函数拟合任务 """
import numpy as np
from matplotlib import pyplot as plt
from Models.net import BPNet
from Plot.styles import mp_seaborn_light
# 生成数据
x = np.arange(0, np.pi, 0.1).reshape(-1, 1)
y = np.sin(x)

# 创建bp神经网络并训练
bp = BPNet(1, [3], 1, "tanh")
bp.train(x, y, 1000, lr=0.01)

new_x = np.arange(0, np.pi, 0.1).reshape(-1, 1)
new_y = np.sin(new_x)
print('\n预测结果:\n', bp.predict(new_x))
print('\n准确率:\n', bp.evaluate(new_x, new_y))

# 绘制图像
plt.style.use(mp_seaborn_light())
bp.plot_network()
bp.plot_data(new_x, new_y)
plt.show()
