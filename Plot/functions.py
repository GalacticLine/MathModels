""" 常用绘图函数 """
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
from matplotlib import pyplot as plt


def plot_matrix(data, title='矩阵热力图', x_rotation='vertical', y_rotation='horizontal', cmap='Blues'):
    """
    绘制矩阵热力图
    :param data: 二维数组
    :param x_rotation: x轴旋转
    :param y_rotation: y轴旋转
    :param cmap: 颜色映射表
    :param title: 绘图标题
    """
    arr = np.asarray(data)
    df = pd.DataFrame(data)

    plt.figure()

    plt.imshow(arr, cmap)

    bar = plt.colorbar()
    bar.outline.set_visible(False)

    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=x_rotation)
    plt.yticks(np.arange(len(df.index)), df.index, rotation=y_rotation)

    for (i, j), val in np.ndenumerate(arr):
        if val > np.mean(arr):
            text_color = 'white'
        else:
            text_color = 'black'
        plt.text(j, i, "%.2f" % val, ha="center", va="center", color=text_color, fontsize=8)

    plt.title(title)
    plt.grid(visible=False)


def plot_fitting(x,
                 y,
                 x_fit,
                 y_fit,
                 title='拟合图',
                 x_label='时间',
                 y_label='y',
                 labels=('原数据', '拟合值'),
                 x_rotation=0,
                 x_ticks_labels=None,
                 latex=None):
    """
    绘制拟合图
    :param x: 自变量x
    :param y: 因变量y
    :param x_fit: 拟合自变量x
    :param y_fit: 拟合因变量y
    :param title: 标题
    :param x_label: x轴标题
    :param y_label: y轴标题
    :param labels: 图例
    :param x_rotation: x轴旋转
    :param x_ticks_labels: x轴刻度标签
    :param latex: latex公式，默认None，即不启用公式
    :return:
    """
    plt.figure()

    if len(x) <= len(x_fit):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.scatter(x, y, label='原数据', s=50, c=colors[1], zorder=10)
    else:
        plt.plot(x, y, "o-", markersize=6, markeredgecolor="white")
    plt.plot(x_fit, y_fit)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(labels=labels)
    plt.xticks(rotation=x_rotation)

    if latex is not None:
        plt.annotate(r'$%s$' % latex,
                     xy=(np.mean(x), np.mean(y)),
                     xytext=(np.mean(x), np.mean(y) + 0.1 * (max(y) - min(y))),
                     ha='center', va='center', fontsize=8)

    if x_ticks_labels is not None:
        plt.xticks(x, x_ticks_labels)

    plt.tight_layout()


def plot_forecast(data,
                  forecast,
                  title='预测图',
                  x_label='x',
                  y_label='y',
                  labels=('原数据', '预测值', '精度误差'),
                  x_rotation=0,
                  x_ticks_labels=None):
    """
    绘制预测图

    :param data: 数据
    :param forecast: 预测数据
    :param title: 标题
    :param x_label: x轴标题
    :param y_label: y轴标题
    :param labels: 图例
    :param x_rotation: x刻度旋转角度
    :param x_ticks_labels: x轴刻度标签
    :return:
    """
    n = len(data)
    total = n + len(forecast)

    plt.figure()
    plt.plot(np.arange(0, n), data, "o-", markersize=6, markeredgecolor="white")
    plt.plot(np.arange(n, total), forecast, "o-", markersize=6, markeredgecolor="white")
    plt.plot([n - 1, n], [data[-1], forecast[0]], ':')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(labels=labels)
    plt.xticks(rotation=x_rotation)
    if x_ticks_labels is not None:
        plt.xticks(np.arange(0, total), x_ticks_labels)
    plt.tight_layout()


def subplot_plot(x_arr, y_arr, title='多子图', x_label='总X轴', y_label='总Y轴', plot_type='plot'):
    """
    绘制多个子图
    :param x_arr:
    :param y_arr:
    :param title:
    :param x_label:
    :param y_label:
    :param plot_type:
    :return:
    """
    plt.figure()
    fig, axes = plt.subplots(x_arr.shape[0], x_arr.shape[1], sharex='all', sharey='all')

    for i in range(x_arr.shape[0]):
        for j in range(x_arr.shape[1]):
            plot_func = getattr(axes[i, j], plot_type)
            plot_func(x_arr[i, j], y_arr[i][j])
    fig.suptitle(title, size=20)
    fig.text(0.5, 0.04, x_label, ha='center')
    fig.text(0.04, 0.5, y_label, va='center', rotation='vertical')


def plot_residuals(y, y_pre, r_title='残差图', x_label='y', alpha=0.95):
    res = np.subtract(y, y_pre)
    interval = stats.t.interval(alpha=alpha,
                                df=len(res) - 1,
                                loc=np.mean(res),
                                scale=stats.sem(res))
    y_err = interval[1] - interval[0]

    plt.figure()
    color = plt.rcParams['axes.prop_cycle']
    color = list(color)[1]['color']
    plt.errorbar(y_pre, res, yerr=y_err, fmt='o', ecolor=color,
                 capsize=6, markersize=8, label='残差', markeredgecolor='white')
    plt.axhline(y=0, color='#666666', zorder=0, linestyle='--')
    plt.title(r_title)
    plt.xlabel(x_label)
    plt.ylabel('残差值')
    plt.legend()


def plot_network(input_size, hidden_sizes, output_size):
    """
    绘制神经网络结构图

    :param input_size: 输入层
    :param hidden_sizes: 隐藏层
    :param output_size: 输出层
    :return:
    """

    max_range = np.max([np.max(input_size), np.max(hidden_sizes), np.max(output_size)])

    def get_nodes(size):
        if max_range == 1:
            return [0]
        if max_range == 2:
            if size > 1:
                return [0, 1]
            else:
                return [0.5]
        return np.linspace(0, max_range - 1, size + 2)[1: - 1]

    input_nodes = get_nodes(input_size)
    output_nodes = get_nodes(output_size)
    hidden_nodes = []

    spaces = np.linspace(1, 3, num=len(hidden_sizes) + 2)[1:- 1]

    line_style = dict(color='k', linewidth=0.1, zorder=0)

    plt.figure()
    plt.scatter([1] * input_size, input_nodes, s=500, label='输入层')

    for space, hidden_size in zip(spaces, hidden_sizes):
        if hidden_size == max_range:
            nodes = range(0, hidden_size)
        else:
            nodes = get_nodes(hidden_size)
        hidden_nodes.append(nodes)

        s = 500 - hidden_size ** 2
        if s < 0:
            s = 20
        plt.scatter([space] * hidden_size, nodes, s=s, label='隐藏层')

    plt.scatter([3] * output_size, output_nodes, s=500, label='输出层')

    for i in input_nodes:
        for j in hidden_nodes[0]:
            plt.plot([1, spaces[0]], [i, j], **line_style)

    count = 0
    for x, y in zip(spaces[:-1], spaces[1:]):
        for i in hidden_nodes[count]:
            for j in hidden_nodes[count + 1]:
                plt.plot([x, y], [i, j], **line_style)
        count += 1

    for i in hidden_nodes[-1]:
        for j in output_nodes:
            plt.plot([spaces[-1], 3], [i, j], **line_style)

    for i, node in enumerate(input_nodes):
        plt.text(0.9, node, f'输入层 {i + 1}', ha='right', va='center')

    for i, space in enumerate(spaces):
        for j, node in enumerate(hidden_nodes[i]):
            plt.text(space + 0.1, node, f'隐藏层 {j + 1}', ha='left', va='center')

    for i, node in enumerate(output_nodes):
        plt.text(3.1, node, f'输出层 {i + 1}', ha='left', va='center')

    plt.axis('off')


def plot_cluster(data, labels, title='聚类散点图', x_label='x', y_label='y', z_label='z'):
    """
    绘制聚类散点图。
    :param data: 原数据
    :param labels: 聚类标签
    :param title: 标题
    :param x_label: x轴标题
    :param y_label: y轴标题
    :param z_label: Z轴标题，当绘制三维图像时启用
    :return:
    """
    un_labels = np.unique(labels)
    n = np.asarray(data).shape[1]

    fig = plt.figure()
    if n == 2:
        for i, label in enumerate(un_labels):
            points = data[labels == label]
            if label == -1:
                text = '噪点'
            else:
                text = f'聚类 {i}'
            plt.scatter(points[:, 0], points[:, 1], label=text, s=50)

    elif n == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(un_labels):
            points = data[labels == label]
            if label == -1:
                text = '噪点'
            else:
                text = f'聚类 {i}'
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=text, s=50)
            ax.set_zlabel(z_label)
    else:
        raise ValueError(f'不支维度为{n}的数据，仅可绘制二维或三维数据图像。')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()


def plot_shortest_path(distances, best_path, title='路径优化'):
    """
    绘制路径优化图
    :param distances: 距离
    :param best_path: 最优距离
    :param title: 标题
    :return:
    """
    graph = nx.Graph()
    distances = np.round(distances, 2)
    # 添加节点
    n_nodes = distances.shape[0]
    node_labels = range(n_nodes)
    graph.add_nodes_from(node_labels)

    # 添加边
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            graph.add_edge(i, j, weight=distances[i, j])

    # 绘制图形
    pos = nx.circular_layout(graph)
    edge_labels = nx.get_edge_attributes(graph, 'weight')

    plt.figure()
    nx.draw_networkx(graph, pos, with_labels=True, node_size=1000, node_color='white', edgecolors='black',
                     edge_color='#AAAAAA')

    # 绘制最优路径
    best_path_edges = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]

    nx.draw_networkx_edges(graph, pos, edgelist=best_path_edges, width=1.5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    # 添加标题
    plt.title(title)
