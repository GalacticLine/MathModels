""" 特征筛选测试 """
import numpy as np
from DataProcess import selector

ary = [[1, 2, 5, 4.5, 1],
       [2, 4, 32, 36, 0],
       [3, 6, 55, 55, 1],
       [4, 8, 55, 50, 0],
       [5, 10, 50, 49.5, 1]]
ary = np.asarray(ary)

# 方差选择法
var_selected = selector.var_select(ary, threshold=5)
var_result = ary[:, var_selected]

# 相关系数法
cor_selected = selector.cor_select(ary, threshold=0.5)
cor_result = ary[:, cor_selected]

# VIF法选择
vif_selected = selector.vif_select(ary)
vif_result = ary[:, vif_selected]

# 卡方选择
chi2_selected = selector.chi2_select(ary, ary[:, 2], 1)
chi2_result = ary[:, chi2_selected]

# 互信息法选择
mutual_selected = selector.mutual_info_select(ary, ary[:, 2], 1)
mutual_result = ary[:, mutual_selected]

pass
