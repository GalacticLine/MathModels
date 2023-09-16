""" 绘图风格包 """
from matplotlib import cycler, pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

mp_seaborn_base = {
    'figure.figsize': (8, 8),
    'figure.dpi': 100,
    'axes.grid': True,
    'axes.labelpad': 14,
    'axes.titlepad': 16,
    'axes.facecolor': "#EDEDF2",
    'axes.titlecolor': '#333333',
    'axes.labelcolor': '#535353',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'ytick.color': '#535353',
    'ytick.left': False,
    'ytick.direction': "in",
    'xtick.color': '#535353',
    'xtick.bottom': False,
    'xtick.direction': "in",
    'grid.color': "white",
    'grid.linewidth': 1,
    'grid.alpha': 0.4,
    'text.color': "535353",
    'legend.edgecolor': 'w',
    'scatter.edgecolors': 'w',
    'patch.force_edgecolor': True,
    'patch.edgecolor': 'white',
    'patch.linewidth': 0.6,
    'boxplot.capprops.color': '#535353',
    'boxplot.boxprops.color': 'white',
    'boxplot.boxprops.linewidth': 0.6,
    'boxplot.whiskerprops.color': '#535353',
    'boxplot.whiskerprops.linewidth': 0.6,
    'boxplot.whiskerprops.linestyle': ':',
    'boxplot.medianprops.color': 'white',
    'boxplot.meanprops.color': 'white',
    'boxplot.meanprops.markeredgecolor': 'white',
    'boxplot.meanprops.markerfacecolor': 'white',
}  # seaborn 风格


def mp_seaborn_light():
    """ seaborn 清新配色 """
    style = mp_seaborn_base.copy()
    style['axes.prop_cycle'] = cycler('color', ['fb9489', 'a9ddd4', '9ec3db', 'cbc7de', 'fdfcc9'])
    return style


def mp_seaborn_autumn():
    """ seaborn 秋季配色 """
    style = mp_seaborn_base.copy()
    style['axes.prop_cycle'] = cycler('color', ['d2b48c', '9b7653', 'a2cffe', 'c8e7ed', 'f0c891'])
    return style


def mp_seaborn_hot():
    """ seaborn 高温配色 """
    style = mp_seaborn_base.copy()
    style['axes.prop_cycle'] = cycler('color', ['FF0000', 'FF4500', 'FF7F00', 'FFA500', 'FFC300', 'FFD700', 'FFFF00'])
    return style


def mp_seaborn_cold():
    """ seaborn 冷色配色 """
    style = mp_seaborn_base.copy()
    style['axes.prop_cycle'] = cycler('color', ['fd98c9', 'fe61ad', 'ad91cb', '7b52ae', '67329f', '6b86ff', '2049ff'])
    return style


def mp_seaborn_green():
    """ seaborn 绿色系配色 """
    style = mp_seaborn_base.copy()
    style['axes.prop_cycle'] = cycler('color', ['fafa00', '9acd32', '00ff7f', '00c5cd', '87ceeb', '008b8b'])
    return style


mp_black_base = {
    'figure.figsize': (8, 8),
    'figure.dpi': 100,
    'figure.edgecolor': 'black',
    'figure.facecolor': 'black',
    'axes.prop_cycle': cycler('color', ['1714ff', '14d7ff', '82ff14', 'ffe014', 'ff7114', 'ff142d', 'ff14bb']),
    'axes.grid': True,
    'axes.labelpad': 14,
    'axes.titlepad': 16,
    'axes.facecolor': 'black',
    'axes.titlecolor': '#999999',
    'axes.labelcolor': '#999999',
    'ytick.color': '#999999',
    'xtick.color': '#999999',
    'ytick.minor.visible': True,
    'xtick.minor.visible': True,
    'grid.color': 'yellow',
    'grid.linewidth': 0.1,
    'lines.linewidth': 1,
    'lines.markersize': 2,
    'text.color': 'white',
    'legend.edgecolor': '#EAEAF2',
    'scatter.edgecolors': 'black',
    'patch.force_edgecolor': True,
    'patch.edgecolor': 'black',
    'patch.linewidth': 0.5,
    'boxplot.capprops.color': 'yellow',
    'boxplot.capprops.linewidth': 0.4,
    'boxplot.boxprops.color': 'black',
    'boxplot.boxprops.linewidth': 0.2,
    'boxplot.whiskerprops.color': 'yellow',
    'boxplot.whiskerprops.linewidth': 0.1,
    'boxplot.medianprops.color': 'black',
    'boxplot.meanprops.color': 'yellow',
    'boxplot.meanprops.markeredgecolor': 'yellow',
    'boxplot.meanprops.markerfacecolor': 'yellow',
    'axes3d.xaxis.panecolor': (0.95, 0.95, 0.95, 0.1),
    'axes3d.yaxis.panecolor': (0.9, 0.9, 0.9, 0.1),
    'axes3d.zaxis.panecolor': (0.925, 0.925, 0.925, 0.1)
}  # 黑底风格

mp_white_base = {
    'figure.figsize': (8, 8),
    'figure.dpi': 100,
    'axes.grid': True,
    'axes.labelpad': 14,
    'axes.titlepad': 16,
    'axes.titlecolor': '#333333',
    'axes.labelcolor': '#535353',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'ytick.color': '#535353',
    'ytick.direction': "in",
    'xtick.color': '#535353',
    'xtick.direction': "in",
    'grid.linewidth': 1,
    "grid.alpha": 0.25,
    'lines.linewidth': 1,
    'lines.markersize': 2,
    'text.color': "535353",
    'legend.edgecolor': 'w',
    'scatter.edgecolors': 'w',
    'patch.force_edgecolor': True,
    'patch.edgecolor': 'white',
    'patch.linewidth': 0.6,
    'boxplot.capprops.color': '#535353',
    'boxplot.boxprops.color': 'white',
    'boxplot.boxprops.linewidth': 0.6,
    'boxplot.whiskerprops.color': '#535353',
    'boxplot.whiskerprops.linewidth': 0.6,
    'boxplot.whiskerprops.linestyle': ':',
    'boxplot.medianprops.color': 'white',
    'boxplot.meanprops.color': 'white',
    'boxplot.meanprops.markeredgecolor': 'white',
    'boxplot.meanprops.markerfacecolor': 'white',
}  # 白底风格


def mp_white_light():
    """ 白底 清新配色 """
    style = mp_white_base.copy()
    style['axes.prop_cycle'] = cycler('color', ['fb9489', 'a9ddd4', '9ec3db', 'cbc7de', 'fdfcc9'])
    return style


def mp_white_grey():
    """ 白底 灰黑配色 """
    style = mp_white_base.copy()
    style['axes.prop_cycle'] = cycler('color', ['333333', '444444', '666666', '888888', 'bbbbbb'])
    return style
