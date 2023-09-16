""" 多准则决策模型 """
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, zscore
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from Plot.functions import plot_matrix
from Tools.numpyHelper import np_reciprocal, np_rank


class AhpData:

    def __init__(self, data, names=None, label: str = None):
        """
        层次分析法单层结构类。
        :param data: 判断矩阵 (非正互反矩阵时，自动按上三角改为正互反矩阵)
        :param names: 准则或方案名集合
        :param label: 标签名
        """
        arr = np_reciprocal(data)
        weight, con = self.fit(arr)

        self.label = label
        self.data = pd.DataFrame(arr, names, names)
        self.weight = pd.Series(weight, names, name='权重')
        self.con = pd.Series(con, name="一致性检验")

    @staticmethod
    def fit(data):
        """
        层次分析法实现。
        :param data: 判断矩阵
        :return: 权重，一致性检验结果
        """
        # 计算特征值和特征向量
        va, vc = np.linalg.eig(data)
        va, vc = va.real, vc.real

        # 获取最大特征值对应的特征向量，并归一化得到权重
        max_vc = vc[:, va.argmax()]
        weight = max_vc / max_vc.sum()

        # 计算一致性指标
        n = va.size
        ci = (np.max(va) - n) / (n - 1)
        ri = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        ri = ri[n - 1]
        cr = ci / ri

        # 构建一致性检验结果
        con = {'CI': ci, 'RI': ri, 'CR': cr, '检验结果': cr < 0.1}

        return weight, con

    @classmethod
    def from_excel(cls, path: str, sheet_name: str | int = 0):
        """
        从Excel中读取矩阵，并生成层次分析法单层结构。
        :param path: 文件路径
        :param sheet_name: 工作表名
        :return: 层结构
        """
        df = pd.read_excel(path, sheet_name)
        empty_row = df[df.isnull().all(axis=1)].index

        if len(empty_row) > 0:
            df = df.head(empty_row[0])
        new_df = df.set_index(df.columns[0])
        new_df.index.name = None

        obj = cls(new_df, new_df.columns, sheet_name)
        return obj

    def to_excel(self, path: str):
        """
        将矩阵，权重，一致性检验结果，导出为指定Excel文件。
        :param path: 文件路径
        :return:
        """
        with pd.ExcelWriter(path) as writer:
            self.to_excel_by_writer(writer)

    def to_excel_by_writer(self, writer: pd.ExcelWriter):
        """
        将矩阵，权重，一致性检验结果，导出为指定Excel文件。
        :param writer: Excel导出器
        :return:
        """
        dfs = self.data, self.weight, self.con

        start_row = 0
        for df in dfs:
            df.to_excel(writer, self.label, startrow=start_row)
            start_row += df.shape[0] + 2

    def plot_matrix(self,
                    title='矩阵热力图',
                    x_rotation='vertical',
                    y_rotation='horizontal',
                    cmap='Blues'):
        """
        绘制层次分析法矩阵热力图。
        :param title: 标题
        :param cmap: 颜色映射表
        :param x_rotation: x轴旋转
        :param y_rotation: y轴旋转
        :return:
        """
        plot_matrix(self.data, title, x_rotation, y_rotation, cmap)

    def print_info(self):
        """
        打印判断矩阵，权重，一致性检验结果。
        :return:
        """
        print(''.center(30, '-'))
        print('判断矩阵:')
        print(self.data)
        print('\n权重:')
        print(self.weight)
        print('\n一致性检验:')
        print(self.con)
        print(''.center(30, '-'))


class Ahp:

    def __init__(self, goal: str, cris: AhpData, sols: list[AhpData]):
        """
        层次分析法完整结构类。
        :param goal: 目标层层名
        :param cris: 准则层数据
        :param sols: 方案层数据集合
        """
        self.goal = goal
        self.cris = cris
        self.sols = sols

        if self.cris.label is None:
            self.cris.label = "准则层"
        for idx, sol in enumerate(self.sols):
            if sol.label is None:
                sol.label = f"方案层-准则{idx + 1}"

        self.__final_sol_weight = None

    @classmethod
    def from_excel(cls, path: str):
        """
        从Excel文件中读取层次分析法的完整结构。
        :param path: 文件路径
        :return: 层次分析法完整结构类实例
        """
        filename = os.path.basename(path)
        goal = os.path.splitext(filename)[0]
        excel = pd.ExcelFile(path)
        datas = [AhpData.from_excel(path, name) for name in excel.sheet_names]
        obj = cls(goal, datas[0], datas[1:])
        return obj

    def to_excel(self, path: str):
        """
        将所有的判断矩阵，权重，一致性检验结果，导出为指定Excel文件。
        :return:
        """
        with pd.ExcelWriter(path) as writer:
            self.cris.to_excel_by_writer(writer)
            for sol in self.sols:
                sol.to_excel_by_writer(writer)

    @property
    def final_sol_weight(self):
        """
        计算最终方案权重。
        :return: 权重
        """
        if self.__final_sol_weight is None:
            sols = self.sols
            cri_weight = self.cris.weight

            sol_weights = np.vstack([sol.weight for sol in sols])
            weight = cri_weight @ sol_weights

            self.__final_sol_weight = pd.Series(weight, sols[0].weight.index, name='最终方案权重')
        return self.__final_sol_weight

    def plot_hierarchy(self):
        """
        绘制层次分析法结构图。
        :return:
        """
        nodes = [[self.goal], self.cris.data.columns, self.sols[0].data.columns]

        plt.figure()
        node_style = dict(bbox=dict(boxstyle=f'round,pad=1', fc='white', ec='black'))
        arrow_style = dict(arrowprops=dict(arrowstyle='<-', color='black'))

        xs = []
        ys = []

        num_layers = len(nodes)
        layers = range(num_layers)

        for i in layers:
            x = np.linspace(0.1, 0.8, len(nodes[i]))
            y = 0.8 - 0.3 * i
            xs.append(x)
            ys.append(y)

        for i in layers:
            layer = nodes[i]
            x_curr = xs[i]
            y_curr = ys[i]

            for j in range(len(layer)):

                node = layer[j]
                if i == 0:
                    plt.annotate(node, xy=(0.45, y_curr), **node_style)
                else:
                    plt.annotate(node, xy=(x_curr[j], y_curr), **node_style)

                if i < num_layers - 1:
                    x_next = xs[i + 1]
                    y_next = ys[i + 1]
                    if i == 0:
                        offset = (x_curr[j] + 0.38, y_curr - 0.06)
                    else:
                        offset = (x_curr[j] + 0.03, y_curr - 0.06)
                    for x_next_curr in x_next:
                        plt.annotate('', xy=offset, xytext=(x_next_curr + 0.03, y_next + 0.06), **arrow_style)

        plt.axis('off')

    def print_info(self):
        """
        打印所有判断矩阵，权重，一致性检验结果以及最终方案权重。
        :return:
        """
        print(f' {self.goal} '.center(50, '='))
        print(f'\n\n{self.cris.label}:')
        self.cris.print_info()
        for sol in self.sols:
            print(f'\n\n{sol.label}:')
            sol.print_info()
        print('\n\n最终方案权重:')
        print(self.final_sol_weight)
        print(''.center(52 + len(self.goal), '='))


class Topsis:
    def __init__(self):
        """
        加权topsis法类。
        """
        self.d_best = None
        self.d_worst = None
        self.scores = None

    def fit(self, data, weights):
        """
        加权topsis实现。
        :param data: 数据
        :param weights: 权重，可由熵权法或其他方法计算。
        :return:
        """

        arr = minmax_scale(data)
        if isinstance(data, pd.DataFrame):
            arr = pd.DataFrame(arr, index=data.index, columns=data.columns)

        arr = weights * arr

        best = np.max(arr, axis=0)
        worst = np.min(arr, axis=0)

        d_best = np.linalg.norm(arr - best, axis=1)
        d_worst = np.linalg.norm(arr - worst, axis=1)

        scores = d_worst / (d_best + d_worst)

        self.d_best = d_best
        self.d_worst = d_worst
        self.scores = scores

    @property
    def info(self):
        df = pd.DataFrame([self.d_best, self.d_worst, self.scores], index=['d+', 'd-', '得分']).T
        df['排名'] = df['得分'].rank(ascending=False)
        df = df.sort_values(by='排名')
        return df

    @staticmethod
    def plot_radar(df, title='评价雷达图'):
        m = df.shape[1]
        angles = [n / m * 2 * np.pi for n in range(m)]
        angles += angles[:1]
        plt.figure()
        plt.subplots(subplot_kw=dict(polar=True))
        for i, row in df.iterrows():
            values = row.tolist()
            values += values[:1]
            plt.plot(angles, values, label=row.name)
            plt.fill(angles, values, alpha=0.1)
        plt.xticks(angles[:-1], df.columns)
        plt.legend(bbox_to_anchor=(0.1, 0.1))
        plt.title(title, y=1.05)



class RSR:
    def __init__(self):
        """
        秩和比综合评价法
        """
        self.data = None
        self.idx = None
        self.prob = None
        self.probit = None
        self.rsr = None
        self.rsr_pol = None

    def fit(self, data, weight=None):
        """
        秩和比综合评价法
        :param data: 数据
        :param weight: 权重，若为None，默认使用熵权法
        """
        data = zscore(data, axis=0)

        if weight is None:
            ew = EntropyWeight()
            ew.fit(data)
            weight = np.asarray(ew.weight)

        rank = np_rank(data)
        rsr = np.sum(rank * weight, axis=1) / rank.shape[0]
        idx = np.argsort(rsr)
        rsr = np.sort(rsr)
        rsr_rank = np_rank(rsr)

        prob = rsr_rank / rsr.shape[0]
        prob[-1] = 1 - 1 / (4 * rsr.shape[0])

        probit = 5 - norm.isf(prob)
        r0 = np.polyfit(probit, rsr, deg=1)
        rsr_pol = np.polyval(r0, probit)

        self.data = data
        self.idx = idx
        self.prob = prob
        self.probit = probit
        self.rsr = rsr
        self.rsr_pol = rsr_pol

    def describe(self, bins=3, path=None):
        """


        :param bins:
        :param path: 导出excel文件的路径，默认为None，即不导出
        :return:
        """
        result = pd.DataFrame({'索引': self.idx, '秩数百分比': self.prob, 'Probit': self.probit, 'RSR': self.rsr,
                               'RSR回归': self.rsr_pol})
        result = result.set_index('索引')
        if isinstance(self.data, pd.DataFrame):
            sorted_df = self.data.iloc[result.index].index
            result = result.set_index(sorted_df)
            result = result.sort_index()
        result['评级'] = pd.qcut(result['RSR回归'], bins, labels=range(bins))

        if path is not None:
            result.to_excel(path)
        return result

    def ols_analysis(self):
        model = sm.OLS(self.rsr, sm.add_constant(self.probit)).fit()
        return model.summary()


class GreyCorr:
    def __init__(self):
        self.rho = None
        self.ksi = None
        self.r = None

    def fit(self, x, y, rho=0.5):
        """
        灰色关联分析实现。
        :param x: 特征序列
        :param y: 关联序列
        :param rho: 分辨系数
        :return: 特征关联度，特征序列关联度
        """
        y = np.reshape(y, (-1, 1))

        # 均值归一化
        x_mean = x / np.mean(x, axis=0)
        y_mean = y / np.mean(y, axis=0)

        # 计算关联系数ksi
        d_abs = np.abs(x_mean - y_mean)
        d_max = rho * np.max(d_abs)
        ksi = (np.min(d_abs) + d_max) / (d_abs + d_max)

        # 计算特征关联度r
        r = np.sum(ksi, axis=0) / ksi.shape[0]

        self.rho = rho
        self.ksi = ksi
        self.r = r

    def plot(self):
        ksi = np.asarray(self.ksi)
        r = self.r
        if isinstance(r, pd.Series):
            x = r.index
        else:
            x = range(ksi.shape[0])

        plt.figure()
        for i in range(ksi.shape[1]):
            plt.plot(x, ksi[i, :], 'o-', markersize=8, markerfacecolor='w')
        plt.xticks(x)
        plt.xlabel('特征序列')
        plt.ylabel('关联系数')
        plt.title('灰色关联度系数图')

        plt.figure()
        plt.bar(x, r)
        for i in range(len(x)):
            plt.text(x[i], r[i], f"{r[i]:.2f}", ha='center', va='bottom')
        plt.xticks(x)
        plt.xlabel('特征序列')
        plt.ylabel('特征关联度')
        plt.title('特征序列关联度图')


class Cv:
    def __init__(self):
        """
        变异系数法类。
        """
        self.std = None
        self.mean = None
        self.cv = None
        self.weight = None

    def fit(self, data, d_dof=1):
        """
        变异系数法实现。
        :param data: 数据
        :param d_dof: 自由度
        :return:
        """
        std = np.std(data, axis=0, ddof=d_dof)
        mean = np.mean(data, axis=0)
        cv = std / mean
        weight = cv / np.sum(cv)

        self.std = std
        self.mean = mean
        self.cv = cv
        self.weight = weight

    def test(self):
        """
        变异系数检测

        如果变异系数大于15%，则特征数据可能有异常值。
        :return: 变异系数大于标准的特征索引
        """
        return self.cv >= 0.15

    @property
    def info(self):
        """
        变异系数法描述性分析。
        :return: 描述信息
        """
        return pd.DataFrame([self.std, self.mean, self.cv, self.weight],
                            index=['标准差', '平均值', '变异系数', '权重'])


class EntropyWeight:
    def __init__(self):
        """
        熵权法类。
        """
        self.entropy = None
        self.weight = None

    def fit(self, data):
        """
        熵权法实现。
        :param data: 特征序列
        :return: 权重
        """
        data = data / np.sum(data, axis=0)
        if isinstance(data, pd.DataFrame):
            entropy = data.apply(self.cal_entropy)
        else:
            entropy = np.apply_along_axis(self.cal_entropy, 0, data)
        entropy_sub = 1 - entropy
        weights = entropy_sub / np.sum(entropy_sub)

        self.entropy = entropy
        self.weight = weights

    @staticmethod
    def cal_entropy(prob):
        """
        计算香农熵。
        :param prob: 概率
        :return: 熵值
        """
        tiny = np.finfo(float).tiny
        prob = np.clip(prob, tiny, 1 - tiny)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    @property
    def info(self):
        return pd.DataFrame({'熵值': self.entropy, '权重': self.weight})
