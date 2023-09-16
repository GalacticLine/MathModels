import matplotlib.pyplot as plt
from Models.mcdm import AhpData, Ahp
from Plot.styles import mp_seaborn_light

# 定义总目标
goal = "总目标"

# 定义准则层名
cris_names = ["准则1", "准则2", "准则3"]

# 定义方案层名
sols_names = ["方案A", "方案B", "方案C"]

# 构造准则层（提供右上三角矩阵数据即可自动补全）
cris = AhpData([[0, 2, 3],
                [0, 0, 2],
                [0, 0, 0]], cris_names)

# 构造方案层-准则1
c1sol = AhpData([[0, 1/3, 1],
                 [0, 0, 2],
                 [0, 0, 0]], sols_names)

# 构造方案层-准则2
c2sol = AhpData([[0, 1, 2],
                 [0, 0, 1],
                 [0, 0, 0]], sols_names)

# 构造方案层-准则3
c3sol = AhpData([[0, 1, 2],
                 [0, 0, 1],
                 [0, 0, 0]], sols_names)

# 组成完整结构
ahp = Ahp(goal, cris, [c1sol, c2sol, c3sol])

# 输出判断矩阵、权重、一致性检验结果
ahp.print_info()

# 绘制相关图像
plt.style.use(mp_seaborn_light())
ahp.plot_hierarchy()
c1sol.plot_matrix()
plt.show()

# 将结果导出excel
ahp.to_excel('ahp_test.xlsx')

# 从excel中读取判断矩阵并自动生成结构
# new_ahp = Ahp.from_excel('ahp_test.xlsx')
