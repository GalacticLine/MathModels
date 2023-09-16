""" 加权Topsis法测试 """
import pandas as pd
from matplotlib import pyplot as plt
from Models.mcdm import Topsis, EntropyWeight
from Plot.styles import mp_white_light

# 定义数据
data = {'产品': ['A类产品', 'B类产品', 'C类产品', 'D类产品'],
        '质量': [7, 8, 9, 3],
        '功能': [6, 5, 9, 6],
        '售后': [9, 8, 8, 5]}
df = pd.DataFrame(data).set_index('产品')

# 加权topsis法，权重使用熵权法计算
model = Topsis()
ew = EntropyWeight()
ew.fit(df)
model.fit(df, ew.weight)
print(model.info)

# 绘制图像
plt.style.use(mp_white_light())
model.plot_radar(df)
plt.show()
