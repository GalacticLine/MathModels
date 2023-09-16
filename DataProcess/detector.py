""" 异常值检测 """
import numpy as np
from scipy.stats import t, zscore


def grubbs_detect(data, alpha=0.05):
    """
    Grubbs（格拉布斯）异常值检验

    格拉布斯的测试基于正态假设，如果计算值大于阈值，就可以拒绝原假设，即认为存在异常值。
    :param data: N,M 特征序列
    :param alpha: 显著性水平
    :return: 异常值索引
    """
    n = len(data)
    n2 = n - 2
    t_value = t.ppf(1 - alpha / (2 * n), n2)
    threshold = (n - 1) * t_value / np.sqrt(n * n2 + n * t_value ** 2)
    z = zscore(data)
    idx = threshold < z
    return idx


def zscore_detect(data, threshold=3.0):
    """
    Z-score 异常值检验

    如果某个数据点的Z-score大于指定倍数标准差，则被认为是异常值。
    :param data: N,M 特征序列
    :param threshold: 阈值（通常为2.5或3.0）
    :return: 异常值索引
    """
    z_score = zscore(data)
    idx = np.abs(z_score) > threshold
    return idx


def mad_detect(data, threshold=3.0):
    """
    Mad 异常值检测

    如果某个数据点的绝对偏差超过阈值，则被认为是异常值。
    :param data: N,M 特征序列
    :param threshold: 阈值（通常为2.5或3.0）
    :return: 异常值索引
    """
    median = np.median(data, axis=0)
    sub = np.subtract(data, median)
    tend = np.abs(sub)
    mad = 0.6745 * tend / np.median(tend, axis=0)
    idx = mad > threshold
    return idx


def iqr_detect(data):
    """
    四分位数检测

    如果数据点小于Q1-1.5IQR或大于Q3+1.5IQR，则被认为是异常值。其中IQR为四分位距。
    :param data: N,M 特征序列
    :return: 异常值索引
    """
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    idx = np.logical_or(data < low, data > high)
    return idx


def three_sigma_detect(data, threshold=3):
    """
    3sigma检验异常值
    :param data: N,M 特征序列
    :param threshold: 阈值
    :return: 异常值索引
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    threshold = threshold * std
    low = mean - threshold
    upper = mean + threshold
    idx = np.logical_or(data < low, data > upper)
    return idx
