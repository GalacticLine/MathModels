""" 秩和比法测试 """
import pandas as pd
from Models.mcdm import RSR

# 定义数据
data = pd.DataFrame({'产前检查率': [99.54, 96.52, 99.36, 92.83, 91.71, 95.35, 96.09, 99.27, 94.76, 84.80],
                     '孕妇死亡率': [60.27, 59.67, 43.91, 58.99, 35.40, 44.71, 49.81, 31.69, 22.91, 81.49],
                     '围产儿死亡率': [16.15, 20.10, 15.60, 17.04, 15.01, 13.93, 17.43, 13.89, 19.87, 23.63]},
                    index=list("ABCDEFGHIJ"), columns=['产前检查率', '孕妇死亡率', '围产儿死亡率'])

# 正向化
data["孕妇死亡率"] = 1 / data["孕妇死亡率"]
data["围产儿死亡率"] = 1 / data["围产儿死亡率"]

# RSR模型
model = RSR()
model.fit(data)
print(model.describe())
print(model.ols_analysis())

