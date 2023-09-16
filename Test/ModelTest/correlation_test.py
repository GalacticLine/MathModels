""" 相关性分析方法的测试 """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataProcess.correlation import Correlation
from Plot.styles import mp_seaborn_light

data = np.array([[1, 2, 3],
                 [2, 3, 2],
                 [3, 2, 1],
                 [3, 2, 2]])
data = pd.DataFrame(data, columns=list('ABC'))

model = Correlation()

plt.style.use(mp_seaborn_light())

model.fit(data, 'Spearman')
model.plot_corr()
model.fit(data, 'Pearson')
model.plot_corr()
model.fit(data, 'Kendall')
model.plot_corr()

plt.show()
