""" 因子分析 """
import numpy as np
from scipy.stats import chi2


def kmo(data):
    """
    KMO 检验用于验证数据是否适合进行因子分析。
    :param data: N,M 特征序列
    :return: 检验结果
    """

    # 计算相关系数矩阵及其逆矩阵
    corr = np.corrcoef(data, rowvar=False)
    corr_inv = np.linalg.inv(corr)

    dia = np.diag(np.diag(corr_inv ** -1))
    air = np.sqrt(dia) @ corr_inv @ np.sqrt(dia)

    # 计算范数平方
    a = np.linalg.norm(air - np.diag(np.diag(air))) ** 2
    b = np.linalg.norm(corr - np.eye(corr.shape[0])) ** 2

    # 计算KMO值
    result = b / (a + b)
    if result > 0.8:
        eva = '优秀'
    elif result > 0.6:
        eva = '合格'
    else:
        eva = '不合格'
    return {'KMO值': result, '评估': eva}


def bartlett(data):
    """
    Bartlett球形检验则用于检验数据的相关性。

    如果 p 值小于设定的显著性水平（如 0.05），则可以拒绝原假设，即认为数据之间存在显著的相关性。
    :param data: 数据
    :return: 统计量，p值
    """

    data = np.asarray(data)
    n, p = data.shape

    # 计算相关系数矩阵及其行列式
    corr = np.corrcoef(data, rowvar=False)
    corr_det = np.linalg.det(corr)

    # 计算统计量
    statistic = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)

    # 计算自由度
    dof = p * (p - 1) / 2

    # 计算p值
    p_value = chi2.sf(statistic, dof)

    return statistic, p_value
