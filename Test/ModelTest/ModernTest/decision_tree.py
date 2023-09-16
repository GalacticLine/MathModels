""" 使用sklearn决策树进行鸢尾花数据集分类任务 """
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# 载入数据集
iris = load_iris()

# 划分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 创建决策树分类器并拟合数据
clf = DecisionTreeClassifier(random_state=42, max_depth=3, max_leaf_nodes=4)
clf.fit(x_train, y_train)

# 预测测试集的结果
y_pre = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pre)
print("准确率：", accuracy)

# 绘制决策树
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6))
iris.feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
iris.target_names = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
