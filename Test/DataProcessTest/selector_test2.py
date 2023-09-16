""" 使用sklearn和statsmodels进行异常值检测 """
import numpy as np
# sklearn.feature_selection 提供了相当多的特征筛选方法
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 特征矩阵
data = np.array([[0, 5, 1, 3],
                 [1, 1, 1, 3],
                 [4, 1, 1, 3],
                 [0, 1, 1.1, 3]])

# 方差选择特征值法
selector = VarianceThreshold(threshold=0.1)
var_selected = selector.fit_transform(data)

# vif选择特征值法
vif = np.array([variance_inflation_factor(data, i) for i in range(data.shape[1])])
vif_selected = data[:, vif < 5]

x, y = data[:, :-1], data[:, -1]
# 卡方选择特征值法
selector = SelectKBest(score_func=chi2, k=2)
chi2_selected = selector.fit_transform(x, y)

# 互信息法筛选特征值
selector = SelectKBest(score_func=mutual_info_classif, k=2)
mut_selected = selector.fit_transform(x, y)

# RFE递归特征消除法筛选特征值
rfe = RFE(LinearRegression(), n_features_to_select=2)
rfe_selected = rfe.fit_transform(x, y)

pass
