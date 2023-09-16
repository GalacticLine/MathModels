""" 使用sklearn进行逻辑回归 """
import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(0)
data0 = np.random.multivariate_normal([2, 3], [[2, 0], [0, 2]], 500)
data1 = np.random.multivariate_normal([8, 7], [[1, 0], [0, 1]], 500)

x = np.concatenate((data0, data1))
y = np.concatenate((np.zeros(500), np.ones(500)))

model = LogisticRegression()

model.fit(x, y)

y_pre = model.predict([[1, 2], [3, 4], [5, 6]])
print(y_pre)
