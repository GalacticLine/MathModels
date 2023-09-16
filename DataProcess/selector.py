""" 特征选择 """
import numpy as np
import statsmodels.api as sm


def var_select(data, threshold=0):
    """
    方差选择法筛选特征序列。

    选择方差超过给定阈值的特征，丢弃方差低于阈值的特征。
    :param data: N,M 特征序列
    :param threshold: 阈值
    :return: 选择结果
    """
    var = np.var(data, axis=0)
    selector = var > threshold
    return selector


def cor_select(data, threshold=0.5):
    """
    相关系数法筛选特征序列。

    计算特征与目标变量之间的相关系数来衡量特征的重要性，并筛选出与目标变量高度相关的特征。
    :param data: N,M 特征序列
    :param threshold: 阈值
    :return: 选择结果
    """
    cor = np.corrcoef(data, rowvar=False)
    cor = np.abs(cor) > threshold
    selector = np.where(np.sum(cor, axis=0) > 1)[0]
    return selector


def vif_select(data, threshold=10.0):
    """
    VIF法筛选特征序列。

    计算每个特征变量的VIF值，判断是否存在多重共线性，
    并将VIF值低于阈值的特征变量选入最终的特征集合中。以此降低多重共线性对模型的影响。
    :param data: N,M 特征序列
    :param threshold: 阈值（VIF值在1-5之间表示存在轻微多重共线性,大于5,则表示存在较严重的多重共线性）
    :return: 选择结果
    """
    data = np.asarray(data)
    n = data.shape[1]
    vif = np.zeros(n)

    for i in range(n):
        y = np.delete(data, i, axis=1)
        x = data[:, i]

        # 最小二乘法计算r方
        model = sm.OLS(x, y)
        r_squared = model.fit().rsquared

        vif[i] = 1.0 / (1.0 - r_squared)

    selector = np.where(vif < threshold)[0]
    return selector


def chi2_select(x, y, n_to_select):
    """
    卡方选择筛选特征序列。

    使用卡方检验来评估特征变量与目标变量之间的独立性假设。
    :param x: 自变量
    :param y: 因变量
    :param n_to_select: 选择特征数
    :return: 选择结果
    """
    x = np.asarray(x)
    n_features = x.shape[1]
    scores = []

    for i in range(n_features):
        obs = np.histogram2d(x[:, i], y, bins=2)[0]
        row_sum = obs.sum(axis=1)
        col_sum = obs.sum(axis=0)
        obs_sum = obs.sum()
        expect = np.outer(row_sum, col_sum) / obs_sum
        chi2 = np.sum((obs - expect) ** 2 / expect)
        scores.append(chi2)

    selector = np.argsort(scores)[-n_to_select:]

    return selector


def mutual_info_select(x, y, n_to_select):
    """
    互信息法筛选特征序列。
    互信息法可用于评估每个特征与目标变量的相关性，进而确定哪些特征对目标变量的预测具有较高的重要性.
    :param x: 自变量
    :param y: 因变量
    :param n_to_select: 选择特征数
    :return: 选择结果
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_features = x.shape[1]
    info_s = []

    def cal_entropy(labels):
        uniq_count = np.unique(labels, return_counts=True)[1]
        uniq_prob = uniq_count / len(labels)
        entropy = -np.sum(uniq_prob * np.log2(uniq_prob))
        return entropy

    for i in range(n_features):
        feature = x[:, i]
        values = np.unique(feature)
        ent1 = cal_entropy(y)

        info = 0
        for val in values:
            y_selected = y[feature == val]
            ent2 = cal_entropy(y_selected)
            info += len(y_selected) / len(y) * ent2

        mutual_info = ent1 - info
        info_s.append(mutual_info)

    selector = np.argsort(info_s)[-n_to_select:]
    return selector
