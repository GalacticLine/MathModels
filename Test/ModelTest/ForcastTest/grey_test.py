""" 灰色预测模型测试 """
import numpy as np
from matplotlib import pyplot as plt
from Models.forecast import GM11
from Plot.styles import mp_seaborn_light

data = np.array([1, 3, 6, 10, 15, 20])

# 灰色预测
model = GM11()
model.fit(data, 3)
print(model.posterior_error())
print(model.small_error())

# 绘制预测图
plt.style.use(mp_seaborn_light())
model.plot_forecast()
model.plot_fitting()
plt.show()
