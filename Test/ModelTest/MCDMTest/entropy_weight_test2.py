""" 熵权法 直接计算熵值权重 """
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy

arr = [[1, 4.5],
       [2, 3],
       [3, 2],
       [4, 1]]
data = pd.DataFrame(arr, columns=['好评', '差评'])

# 正向化
data['差评'] = data['差评'].max() - data['差评']

# 标准化
data = minmax_scale(data)

# scipy已经提供了熵值的计算方法，可以直接调用。

# entropies = data.apply(lambda x: entropy(x / np.sum(x), base=2))  # pandas
entropies = np.apply_along_axis(lambda x: entropy(x / np.sum(x), base=2), 0, data)  # numpy

# 计算权重
weights = (1 - entropies) / np.sum(1 - entropies)

print(weights)
