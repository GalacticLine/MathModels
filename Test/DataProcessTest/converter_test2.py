""" 使用scipy和sklearn进行数据变换 """
import numpy as np
from scipy import stats
# sklearn.preprocessing 提供了相当多的标准化方法
from sklearn.preprocessing import minmax_scale, StandardScaler

np.random.seed(0)
arr = np.random.randint(1, 10, (4, 3))

# zscore标准化
zs_arr = stats.zscore(arr)

# 最小最大标准化
minmax_arr = minmax_scale(arr)

# zscore标准化 sklearn
data = StandardScaler().fit_transform(arr)

pass
