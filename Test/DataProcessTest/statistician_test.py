""" 统计描述测试 """
import numpy as np
import pandas as pd
from DataProcess import statistician as st

df = pd.DataFrame(np.random.randint(0, 5, (6, 4)))

des = st.describe_statistic(df, is_translate=True)
fre = st.frequency_statistic(df)

print('描述性统计：')
print(des)

print('频数统计：')
print(fre)
