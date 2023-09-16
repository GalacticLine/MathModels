""" 蒙特卡洛测试 """
from matplotlib import pyplot as plt
from Models.monteCarlo import MonteCarlo
from Plot.styles import mp_seaborn_light

mc = MonteCarlo()


def masker(x, y):
    return x ** 2 + y ** 2 - 1 <= 0


plt.style.use(mp_seaborn_light())

mc.fit2d(-1, 1, -1, 1, 10000, masker)
print('面积近似值:', mc.approx)
mc.plot_mc()


def masker(x, y, z):
    return x ** 2 + y ** 2 <= z**2


mc.fit3d(-1, 1, -1, 1, -1, 1, 10000, masker)
print('体积近似值:', mc.approx)
mc.plot_mc()

plt.show()

