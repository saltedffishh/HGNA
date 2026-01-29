import pandas as pd
import os

# 1. 获取当前脚本所在的目录 (即 .../na_cells/src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (即 .../na_cells)，也就是当前目录的上一级
project_root = os.path.dirname(current_dir)

# 3. 拼接数据的完整路径
data_path_expr= os.path.join(project_root, "COVID19_data", "Asymptomatic_scaledata.txt")
data_path_meta= os.path.join(project_root, "COVID19_data", "Asymptomatic_metadata.csv")
print("正在读取文件:")


def load_expression_matrix(path):
    expr = pd.read_csv(
        path,
        sep=r"\s+",
        index_col=0
    )
    return expr

import pandas as pd

def load_cell_metadata(path):
    meta = pd.read_csv(path, index_col=0)
    return meta

expr= load_expression_matrix(data_path_expr)
meta= load_cell_metadata(data_path_meta)
# print(expr.shape)
# print(expr.index[:5])
# print(expr.columns[:5])

print("------ 数据检查报告 ------")
print(f"1. 维度检查: {expr.shape} (行=基因数, 列=细胞数)")
if expr.shape[1] == 1:
    print("   ❌ 警告：列数为 1，可能是分隔符 sep 设置错误！")

print("\n2. 内容预览 (前3行3列):")
print(expr.iloc[:3, :3])

print("\n3. 索引示例 (基因名):", expr.index[:3].tolist())
print("4. 列名示例 (细胞ID):", expr.columns[:3].tolist())

print("\n5. 空值检查:", "存在空值" if expr.isnull().values.any() else "无空值 (Pass)")
print("------------------------")

print("meta检查")
print(meta.shape)
print(meta.head())
print(meta.index[:5])
print(meta.columns)
print("------------------------")
print("输出meta")
print(meta)