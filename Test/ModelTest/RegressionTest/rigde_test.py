""" 岭回归测试 """
import matplotlib.pyplot as plt
from Models.regression import RidgeRegression
from Plot.styles import mp_seaborn_light

X = [[1, 2, 2],
     [3, 4, 3],
     [3, 4, 4],
     [5, 2, 3]]
y = [2, 4, 6, 1]

rg = RidgeRegression()
rg.fit(X, y, alpha=0.1)
print(rg.predict(X))

plt.style.use(mp_seaborn_light())
rg.plot_coefficients()
plt.show()
