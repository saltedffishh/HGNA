import argparse
import os
import glob
import pandas as pd
import sys
import re

def get_project_root():
    """获取项目根目录"""
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    return os.path.dirname(src_dir)

def natural_sort_key(filepath):
    """
    自定义排序规则：
    1. 'Asymptomatic' 排在最前面 (-1)
    2. 其他文件按文件名中的第一个数字大小排序
    """
    filename = os.path.basename(filepath)
    
    # 特殊处理：Asymptomatic 视为最优先
    if "Asymptomatic" in filename:
        return -1
    
    # 使用正则表达式提取文件名里的第一个连续数字
    # 例如 'Days_1_4...' -> 提取出 1
    # 'Days_14...' -> 提取出 14
    numbers = re.findall(r'\d+', filename)
    
    if numbers:
        return int(numbers[0]) # 返回数字用于比较
    else:
        return 999 # 如果没数字，排在最后
        
def load_dataset_pairs(dataset_name):
    """
    读取指定文件夹下的 matrix(.txt) 和 metadata(.csv)
    返回: (expr_list, meta_list, filenames)
    """
    root_dir = get_project_root()
    
    # 1. 路径匹配
    possible_paths = [
        os.path.join(root_dir, dataset_name),
        os.path.join(root_dir, f"{dataset_name}_data")
    ]
    target_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
    if not target_path:
        print(f"❌ 错误: 未找到目录 '{dataset_name}'")
        sys.exit(1)

    # 2. 搜索 txt 文件
    search_pattern = os.path.join(target_path, "*_scaledata.txt")
    txt_files = glob.glob(search_pattern)

    if not txt_files:
        print(f"❌ 错误: {target_path} 中没有找到 *_scaledata.txt")
        sys.exit(1)

    # 3. 关键步骤：应用自定义排序 (数字大小排序)
    # 这会把 [Days_14, Days_5] 变成 [Days_5, Days_14]
    txt_files.sort(key=natural_sort_key)

    print(f"📂 正在从目录加载数据: {os.path.basename(target_path)}")
    print(f"   排序策略: Asymptomatic -> 数字从小到大")

    expr_list = [] # 存放表达矩阵
    meta_list = [] # 存放元数据
    file_names = []

    # 4. 循环读取文件对
    for txt_path in txt_files:
        base_name = os.path.basename(txt_path)
        
        # 4.1 推断对应的 CSV 路径
        # 假设规则：XXX_scaledata.txt 对应 XXX_metadata.csv
        csv_name = base_name.replace("_scaledata.txt", "_metadata.csv")
        csv_path = os.path.join(target_path, csv_name)
        
        print(f"   ⏳ 正在读取组: {base_name.split('_scaledata')[0]} ...", end="", flush=True)

        # 4.2 读取 TXT (表达矩阵)
        # 使用你确认过的参数: sep=\s+
        expr = pd.read_csv(txt_path, sep=r"\s+", index_col=0)
        
        # 4.3 读取 CSV (元数据)
        # 检查 csv 是否存在
        if os.path.exists(csv_path):
            # 使用你提供的 csv 读取代码: 默认 sep (逗号), index_col=0
            meta = pd.read_csv(csv_path, index_col=0)
        else:
            print(f"\n   ⚠️ 警告: 找不到对应的元数据 {csv_name}，该位置填充为 None")
            meta = None
            
        # 4.4 存入列表
        expr_list.append(expr)
        meta_list.append(meta)
        file_names.append(base_name)
        
        print(f" ✅ (Expr: {expr.shape}, Meta: {meta.shape if meta is not None else 'Missing'})")

    return expr_list, meta_list, file_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, required=True, help='数据集名称')
    args = parser.parse_args()
    
    # 获取两个列表
    expr_data, meta_data, names = load_dataset_pairs(args.experiment)
    
    # 验证排序结果
    print("\n🔍 最终加载顺序验证:")
    for i, name in enumerate(names):
        print(f"   [{i}] {name}")
        
    # 使用示例
    print("\n💡 调用示例:")
    print("   expr_data[0] 是第一个时间点的表达矩阵")
    print("   meta_data[0] 是第一个时间点的元数据")