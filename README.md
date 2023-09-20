# 数学模型库
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)

储存了常用的一些数学模型以及其他工具，共包含以下4个部分：数据分析及处理、数学模型、绘图、辅助工具。

** 鉴于本人对一些算法的实现过程了解有限，可能无法确保所有算法的正确性，如有错误之处，还望批评指正。

## 介绍


### [数据分析及处理](DataProcess)
包含一些用于数据分析和处理的函数和类。
* 数据变换
  * 标准化
    * 总和标准化
    * 小数定标标准化
    * 对数标准化
    * 最小-最大标准化
    * Z-score标准化
    * 四分位数标准化
  * 中心化
    * 均值中心化
    * 中位数中心化
  * 正向化
    * 极小型指标正向化
    * 中间型指标正向化
    * 区间型指标正向化
  * 二值化
* 数据降维
  * 主成分分析 PCA 
  * 核主成分分析 KernelPCA 
* 异常值检测
  * Grubbs 异常值检验 
  * Z-score 异常值检验 
  * Mad 异常值检测 
  * 四分位数异常值检测 
  * 3sigma 异常值检验 
* 特征筛选
  * 方差法 
  * 相关系数法 
  * VIF法 
  * 卡方法 
  * 互信息法
* 相关性分析
  * Pearson相关系数 
  * Spearman相关系数 
  * Kendall相关系数
  
### [数学模型](Models)
包含常见的数学模型，如回归模型、聚类模型、预测模型等。
* 数据拟合
  * 线性拟合
  * 多项式拟合
  * 曲线拟合
* 多准则决策模型
  * AHP 层次分析法
  * 变异系数法
  * 熵权法
  * RSR 秩和比综合评价法
  * 加权Topsis法
  * 灰色关联分析
* 预测模型
  * 灰色预测模型 GM(1,1)
  * ARIMA 时间预测模型
  * 马尔可夫链预测
* 蒙特卡洛
* 规划模型
  * 线性规划
* 回归模型
  * 线性回归
  * 逻辑回归
  * 岭回归
* 聚类模型
  * KMeans聚类
  * DBSCAN聚类
  * 层次聚类
* 现代模型
  * BP神经网络
  * 遗传算法
  * 退火算法
  * 蚁群算法
  * 粒子群算法
  * 元胞自动机

### [绘图](Plot)
包含一些绘图风格文件和绘图函数。
* 绘图风格包
* Kmeans聚类取色器
* 常用绘图函数
![图集1](https://github.com/Bomb-Cat/MathModels/assets/128875843/edeab394-786f-409a-8ff3-5e55ed945950)

### [辅助工具](Tools)
包含如生成三线表等功能，用于辅助数学建模和数据处理。
* Excel辅助
* 矩阵辅助

### [测试](Test)
包含了一些调用本库中的模型具体使用方法和使用其他库的实现模型的例子。(可删除，不会影响其它包的正常使用)

除了上述模型测试外，还有以下内容：
* 插值
* 非线性规划
* 支持向量机SVM
* 决策树

### 运行依赖
本仓库依赖于Python3.x和以下的Python库：

* [numpy](https://github.com/numpy/numpy)
* [pandas](https://github.com/pandas-dev/pandas)
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [scipy](https://github.com/scipy/scipy)
* [statsmodels](https://github.com/statsmodels/statsmodels)
* [sympy](https://github.com/sympy/sympy)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [networkx](https://github.com/networkx/networkx)

以上库也是科学计算和数据分析和处理中常用的十分强大的工具。(额外推荐 [latex2mathml](https://github.com/roniemartinez/latex2mathml)，这个库可以快速的把latex公式转成mathml格式，方便复制粘贴到word中。)


** 本仓库中部分的模型，在上述库（如scikit-learn）中已经提供了现成的类或函数，且更为精确、完善且成熟。

** 数学建模竞赛代码参与查重，本仓库中的代码，在比赛中请不要直接使用。
