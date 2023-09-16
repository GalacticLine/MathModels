""" 灰色关联分析 """
import pandas as pd
import matplotlib.pyplot as plt
from Models.mcdm import GreyCorr
from Plot.styles import mp_seaborn_light

# 构建数据
x = [[1, 2, 1, 2],
     [2, 4, 1, 2],
     [3, 7, 1, 2],
     [4, 8, 1, 2]]
y = [1, 2, 3, 4]
x = pd.DataFrame(x, columns=list('ABCD'))

# 灰色关联分析
model = GreyCorr()
model.fit(x, y)
print(model.ksi)
print(model.r)

# 绘制图像
plt.style.use(mp_seaborn_light())
model.plot()
plt.show()
