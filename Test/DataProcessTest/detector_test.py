""" 异常值检测测试 """
import numpy as np
from DataProcess import detector

np.random.seed(0)
data = np.random.normal(loc=0.0, scale=1, size=(30, 3))

# 插入异常值
data[5, 1] = 4

# Grubbs检验
gb_idx = detector.grubbs_detect(data)
gb_result = data[gb_idx]

# Z-score检验
zs_idx = detector.zscore_detect(data)
zs_result = data[zs_idx]

# MAD检验
mad_idx = detector.mad_detect(data)
mad_result = data[mad_idx]

# IQR检验
iqr_idx = detector.iqr_detect(data)
iqr_result = data[iqr_idx]

# 3sigma检验
sigma_idx = detector.three_sigma_detect(data)
sigma_result = data[sigma_idx]

pass
