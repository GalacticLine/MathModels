""" 使用sklearn进行岭回归 """
from sklearn.linear_model import Ridge
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练岭回归模型
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)

# 进行预测
y_pre = ridge.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pre)
print("均方误差（MSE）：", mse)
