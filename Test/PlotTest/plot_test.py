""" 绘图风格测试 """
import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, hex2color
from Plot import styles

with plt.style.context(styles.mp_seaborn_light()):
    np.random.seed(0)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    rgb_colors = [hex2color(color) for color in colors]

    plt.figure(figsize=(16, 7))
    plt.subplot(241)
    x = np.linspace(0, np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(2 * x)
    y4 = np.cos(2 * x)
    y5 = np.sin(x + np.pi / 2)
    plt.ylim(-1, 1.2)
    plt.margins(0, 0)
    plt.plot(x, y1, label='Sin(x)')
    plt.plot(x, y2, label='Cos(x)')
    plt.plot(x, y3, label='Sin(2x)')
    plt.plot(x, y4, label='Cos(2x)')
    plt.plot(x, y5, label=f'Sin(x+{chr(960)}/2)')
    plt.fill_between(x, -1, y1, alpha=0.05)
    plt.fill_between(x, -1, y2, alpha=0.1)
    plt.fill_between(x, -1, y3, alpha=0.15)
    plt.fill_between(x, -1, y4, alpha=0.2)
    plt.fill_between(x, -1, y5, alpha=0.25)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('曲线图')
    plt.legend()

    plt.subplot(242)
    sizes = [15, 30, 45, 20]
    labels = ['A', 'B', 'C', 'D']
    patches, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    for autotext in autotexts:
        autotext.set_color('white')
    plt.title('饼图')

    plt.subplot(243)
    data = np.random.randint(0, 10, 100)
    plt.hist(data)
    plt.title('直方图')
    plt.xlabel('数据')
    plt.ylabel('频率')

    plt.subplot(244)
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)
    distances = np.sqrt(x ** 2 + y ** 2)
    normal_distances = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    cmap = LinearSegmentedColormap.from_list('seaborn', rgb_colors)
    plt.scatter(x, y, s=30, alpha=normal_distances, c=distances, cmap=cmap)
    plt.title('散点图')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(245)
    labels = ['A', 'B', 'C', 'D']
    values = [10, 15, 7, 12]
    plt.barh(labels, values, color=colors[0:len(labels)])
    plt.title('条形图')
    plt.xlabel('数据')
    plt.ylabel('类别')

    plt.subplot(246)
    data = np.random.randint(0, 10, (3, 10))
    bp = plt.boxplot(data, patch_artist=True)
    colors_cycle = itertools.cycle(colors)
    for patch in bp['boxes']:
        color = next(colors_cycle)
        patch.set(facecolor=color)
    plt.title('箱线图')
    plt.xlabel('数据')

    plt.subplot(247)
    new_colors = []
    for color in rgb_colors:
        new_colors.append((color[0] * 1.15, color[1] * 1.15, color[2] * 1.15))
    new_cmap = LinearSegmentedColormap.from_list('new_seaborn_cmap', new_colors)
    data = np.random.random((10, 10))
    plt.imshow(data, cmap=new_cmap)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('热力图')

    plt.subplot(248)
    x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    z = np.sin(np.sqrt(x ** 2 + y ** 2))
    plt.contourf(x, y, z, levels=20, cmap=new_cmap.reversed())
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('等高线图')
    plt.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap=cmap.reversed())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D 模拟图')

plt.show()
