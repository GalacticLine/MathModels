""" 使用sklearn进行线性回归 """
import numpy as np
from sklearn.linear_model import LinearRegression

# 创建数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# 创建线性回归模型，并进行训练
model = LinearRegression()
model.fit(X, y)

# 预测
X_new = np.array([[6]])
y_pre = model.predict(X_new)

print("预测结果：", y_pre)
