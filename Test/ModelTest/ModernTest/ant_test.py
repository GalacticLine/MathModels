""" 蚁群算法测试 """
from matplotlib import pyplot as plt
from Models.antcolony import AntColony
from sklearn.metrics import euclidean_distances
from Plot.styles import mp_seaborn_light
from Plot.functions import plot_shortest_path

# 生成随机距离矩阵
distances = euclidean_distances([[0, 1], [1, 2], [3, 6], [5, 15], [1, 3], [2, 7], [2, 2]])
print('距离矩阵:')
print(distances)

aco = AntColony(n_ants=20,
                n_iters=100,
                pheromone_weight=1,
                inspire_weight=5,
                evaporate_rate=0.5)
best_path, shortest_distance = aco.optimize(distances)
print("最佳路径:", best_path)
print("最短距离:", shortest_distance)

# 绘制图像
plt.style.use(mp_seaborn_light())
plot_shortest_path(distances, best_path, title='蚁群算法路径优化')
plt.show()
