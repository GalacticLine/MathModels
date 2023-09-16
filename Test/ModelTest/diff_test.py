""" 偏微分方程测试 """
import matplotlib.pyplot as plt
from Models.diffEquation import SEIRD, Lorenz, LogisticPopulation
from Plot.styles import mp_seaborn_light

plt.style.use(mp_seaborn_light())

seird = SEIRD()
seird.plot_model()

lor = Lorenz()
lor.plot_model()

lp = LogisticPopulation()
lp.plot_model()

plt.show()
