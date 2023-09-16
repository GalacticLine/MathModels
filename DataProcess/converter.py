""" 数据变换 """
import numpy as np


def sum_normalize(data):
    """
    总和标准化

    将每个样本的数据除以特征数据的总和，将数据映射0和1之间。
    :param data: N,M 特征序列
    :return: 标准化结果
    """
    return data / np.sum(data, axis=0)


def decimal_normalize(data):
    """
    小数定标标准化

    通过移动数据的小数点位置，将数据映射0和1之间。
    :param data: N,M 特征序列
    :return: 标准化结果
    """
    data_max = np.max(np.abs(data), axis=0)

    # 计算移动的小数位数并变换数据
    digit = np.ceil(np.log10(data_max))
    data = data / (10 ** digit)
    return data


def min_max_normalize(data, min_value=0, max_value=1):
    """
    最小-最大标准化

    通过对数据进行线性变换，使得数据落入指定的范围内，通常情况下，将数据映射0和1之间。
    :param data: N,M 特征序列
    :param min_value: 最小值
    :param max_value: 最大值
    :return: 标准化结果
    """
    assert max_value > min_value
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data = (data - data_min) / (data_max - data_min)
    data = data * (max_value - min_value) + min_value
    return data


def zscore_normalize(data, d_dof=0):
    """
    Z-score标准化（零-均值标准化）

    通过减去数据的均值然后除以数据的标准差，将数据转换为以0为均值，1为标准差的标准正态分布数据。
    :param data: N,M 特征序列
    :param d_dof: 自由度
    :return: 标准化结果
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=d_dof)
    data = (data - mean) / std
    return data


def log_normalize(data):
    """
    对数标准化

    通过对数运算，将数据转换为以0为均值，1为标准差的标准正态分布数据。
    :param data: N,M 特征序列
    :return: 标准化结果
    """
    data_log = np.log(data)
    mean = np.mean(data_log, axis=0)
    std = np.std(data_log, axis=0)
    data = (data_log - mean) / std
    return data


def quartile_normalize(data, r_tol=1e-8):
    """
    四分位数标准化

    一种常用的数据标准化方法。使用数据的四分位数来进行缩放和平移，将数据转换为具有标准差为1的分布。
    :param data: N,M 特征序列
    :param r_tol: 微小值
    :return: 标准化结果
    """
    # 计算四分位距
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = r_tol

    med = np.median(data, axis=0)
    data = (data - med) / iqr
    return data


def mean_centering(data):
    """
    均值中心化

    通过减去数据的均值，使数据以零为中心。
    :param data: N,M 特征序列
    :return: 标准化结果
    """
    mean = np.mean(data, axis=0)
    data = data - mean
    return data


def median_centering(data):
    """
    中位数中心化

    通过减去数据的中位数，使数据以零为中心。
    :param data: N,M 特征序列
    :return: 标准化结果
    """
    med = np.median(data, axis=0)
    data = data - med
    return data


def neg_positive(data, mode=0):
    """
    极小型指标正向化

    :param data: N,M 极小型特征序列
    :param mode: 默认为0，即使用最大值处理，若为1，使用倒数处理
    :return: 极大型特征序列
    """
    if mode == 0:
        data = np.max(data, axis=0) - data
    elif mode == 1:
        data = 1 / data
    else:
        raise ValueError(f'模式{mode}不存在')
    return data


def median_positive(data, med=None):
    """
    中间型指标正向化
    :param data:  N,M 中间型特征序列
    :param med: 中间值，若为None，默认使用中位数
    :return: 极大型特征序列
    """
    if med is None:
        med = np.median(data, axis=0)
    sub = np.abs(data - med)
    data = 1 - sub / np.max(sub, axis=0)
    return data


def interval_positive(data, left=None, right=None):
    """
    区间型指标正向化
    :param data: N,M 特征序列
    :param left: 左区间，若为None，使用下四分位数作为左区间。
    :param right: 右区间，若为None，使用上四分位数作为右区间。
    :return: 极大型特征序列
    """
    if left is None:
        left = np.percentile(data, 25, axis=0)
    if right is None:
        right = np.percentile(data, 75, axis=0)

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_max = np.max([left - data_min, data_max - right], axis=0)

    data = np.where(data < left, 1 - (left - data) / data_max, data)
    data = np.where((data > left) & (data < right), 1, data)
    data = np.where(data > right, 1 - (data - right) / data_max, data)
    return data


def binary(data, threshold):
    """
    二值化

    将小于阈值的数据转为0，大于阈值的数据转为1。
    :param data: N,M 特征序列
    :param threshold: 阈值
    :return: 二值化结果
    """
    arr = np.asarray(data)
    arr = np.where(arr <= threshold, 0, 1)
    return arr
