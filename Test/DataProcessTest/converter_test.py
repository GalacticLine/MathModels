""" 数据变换测试 """
import numpy as np
import pandas as pd
from DataProcess import converter as st

np.random.seed(0)
arr = np.random.randint(1, 10, (4, 3))

# 总和标准化
sum_result = st.sum_normalize(arr)
sum_test = np.sum(sum_result, axis=0)

# z-score 标准化
zs_result = st.zscore_normalize(arr)
zs_test = [np.mean(zs_result), np.std(zs_result)]

# min-max 标准化
min_max_result = st.min_max_normalize(arr)
min_max_test = [np.min(min_max_result), np.max(min_max_result)]

# 小数定标标准化
dn_result = st.decimal_normalize(arr)

# log变换标准化
log_result = st.log_normalize(arr)
log_test = [np.round(np.mean(log_result)), np.std(log_result)]

# 四分位数标准化
iqr_result = st.quartile_normalize(arr)

# 负向指标正向化
eva_df = pd.DataFrame({'好评数': [1, 3, 4, 2], '差评数': [1, 3, 4, 2]})
eva_df['差评数'] = st.neg_positive(eva_df['差评数'])

# 中间型指标正向化
ph_df = pd.DataFrame({'ph值': [5, 6, 7, 8, 9]})
mp_result = st.median_positive(ph_df, 7)

# 区间型指标正向化
temperature = np.random.normal(36.5, 0.5, 5)
inp_result = st.interval_positive(temperature, 36, 37)

pass
