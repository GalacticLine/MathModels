""" 矩阵运算 """
import numpy as np


def np_symmetric(data):
    """
    将矩阵按上三角修改为对称矩阵。
    :param data: 矩阵
    :return: 正互反矩阵
    """
    arr = np.asfarray(data)
    idx_u = np.triu_indices_from(arr, k=1)
    idx_l = np.tril_indices_from(arr, k=-1)
    arr[idx_l] = arr[idx_u]
    return arr


def np_reciprocal(data):
    """
    将矩阵按上三角修改为正互反矩阵。
    :param data: 矩阵
    :return: 正互反矩阵
    """
    arr = np.asfarray(data)
    if not np.all(arr * arr.T == 1):
        idx_u = np.triu_indices_from(arr, k=1)
        idx_l = np.tril_indices_from(arr, k=-1)
        arr[idx_l] = 1 / arr[idx_u]
        np.fill_diagonal(arr, 1)
    return arr


def np_rank(data):
    """
    numpy 为每个元素分配一个排名值，根据元素的大小或其他指定的排序标准进行排序
    :param data: 数据
    :return: 排名
    """
    return np.argsort(np.argsort(data, axis=0), axis=0) + 1


def np_mode(data):
    """
    numpy 求众数
    :param data: 数据
    :return: 众数
    """
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]
    return mode


