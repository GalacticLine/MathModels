""" 统计描述 """
import pandas as pd
from Tools.excelHelper import marge_excel_col


def describe_statistic(df: pd.DataFrame, is_translate: bool = False, path: str = None):
    """
    更详细的描述性统计。
    :param df: 原数据
    :param is_translate: 是否使用中文索引
    :param path: 导出的excel文件路径，默认为None，即不导出
    :return: 定类和定量的描述性统计表
    """
    obj_cols = ['object', 'category']
    obj = df.select_dtypes(include=obj_cols)
    num = df.select_dtypes(exclude=obj_cols)
    obj_des = num_des = None
    has_obj = obj.shape[1] > 0
    has_num = num.shape[1] > 0

    if has_obj:
        des = obj.describe(include='all')
        des.loc['miss'] = obj.isnull().sum()
        obj_des = des
        if is_translate:
            obj_des.index = ['总数量', '类数', '最多类', '最大频率', '缺失数']

    if has_num:
        des = num.describe()
        des.loc['range'] = des.loc['max'] - des.loc['min']
        des.loc['cv'] = des.loc['std'] / des.loc['mean']
        des.loc['var'] = des.loc['std'] ** 2
        des.loc['kurt'] = num.kurt(numeric_only=True)
        des.loc['skew'] = num.skew(numeric_only=True)
        des.loc['miss'] = num.isnull().sum()
        num_des = des
        if is_translate:
            num_des.index = ['总数量', '平均数', '标准差', '最小值', '25%分位数', '50%分位数', '75%分位数', '最大值',
                             '极差值', '变异值', '方差值', '偏度值', '峰度值', '缺失数']

    if path is not None:
        writer = pd.ExcelWriter(path)
        if has_obj:
            obj_des.to_excel(writer, sheet_name='定类', startrow=1)
        if has_num:
            num_des.to_excel(writer, sheet_name='定量', startrow=1)
        writer.close()

    if has_obj and has_num:
        return obj_des, num_des
    elif has_obj:
        return obj_des
    elif has_num:
        return num_des
    else:
        return None


def frequency_statistic(df: pd.DataFrame, path: str = None):
    """
    频数统计。
    :param df: 原数据
    :param path: 导出的Excel文件路径，默认为None，即不导出
    :return: 频数统计表
    """

    result = pd.concat([pd.DataFrame({
        '特征': x,
        '频数': df[x].value_counts(),
        '频率': df[x].value_counts(normalize=True)
    }) for x in df.columns])

    result.reset_index(inplace=True, names=['样本值'])
    result.set_index('特征', inplace=True)

    if path is not None:
        with pd.ExcelWriter(path) as writer:
            result.to_excel(writer, sheet_name='频数分析', startrow=1)
        marge_excel_col(path)
    return result
