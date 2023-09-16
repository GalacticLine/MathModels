import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Models.forecast import ArimaForcast
from Plot.styles import mp_seaborn_light
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# 创建时间序列
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', end='2026-01-01', freq='M')
ar = np.array([1, -.75, .25])
ma = np.array([1, .65, .35])
arima_process = sm.tsa.ArmaProcess(ar, ma)
data = arima_process.generate_sample(nsample=len(dates))
data = pd.Series(data, index=dates)

# 输出结果
model = ArimaForcast()
model.fit(data, 10, (2, 0, 2))
print(model.forecast)
print('平稳性检验:', model.is_stable)
print('LB检验:', model.not_white_noise)
print(model.best_pdq)

# 绘制图像
plt.style.use(mp_seaborn_light())
plot_acf(data, title='自相关图')
plot_pacf(data, title='偏自相关图')

dates = pd.concat([model.forecast, data], axis=1).index
formatted_dates = [date.strftime("%Y-%m-%d") for date in dates]
model.plot_forecast(x_rotation=45, x_ticks_labels=formatted_dates)

formatted_dates = [date.strftime("%Y-%m-%d") for date in data.index]
model.plot_fitting(x_rotation=45, x_ticks_labels=formatted_dates)
plt.show()
