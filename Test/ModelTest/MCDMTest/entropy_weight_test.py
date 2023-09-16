""" 熵权法测试 """
import pandas as pd
from sklearn.preprocessing import minmax_scale
from Models.mcdm import EntropyWeight

arr = [[1, 4.5],
       [2, 3],
       [3, 2],
       [4, 1]]
data = pd.DataFrame(arr, columns=['好评', '差评'])

# 正向化
data['差评'] = data['差评'].max() - data['差评']

# 标准化
data = minmax_scale(data)

# 熵权法
model = EntropyWeight()
model.fit(data)

print(model.info)
