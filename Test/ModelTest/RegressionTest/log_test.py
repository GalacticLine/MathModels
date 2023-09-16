""" 逻辑回归测试 """
import matplotlib.pyplot as plt
import numpy as np
from Models.regression import LogisticRegression
from Plot.styles import mp_seaborn_light

np.random.seed(0)
data0 = np.random.multivariate_normal([2, 3], [[2, 0], [0, 2]], 500)
data1 = np.random.multivariate_normal([8, 7], [[1, 0], [0, 1]], 500)

x = np.concatenate((data0, data1))
y = np.concatenate((np.zeros(500), np.ones(500)))

lg = LogisticRegression()
lg.fit(x, y)

X_new = [[1, 2], [3, 4], [5, 6]]
y_pre = lg.predict(X_new)
print(y_pre)

plt.style.use(mp_seaborn_light())
lg.plot_logist_r()
plt.show()
