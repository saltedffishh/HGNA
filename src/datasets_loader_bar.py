import argparse
import os
import glob
import pandas as pd
import sys
import re
from tqdm import tqdm

def get_project_root():
    """
    获取项目根目录 (假设脚本位于 src/ 下，项目根目录为上一级)
    """
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    return os.path.dirname(src_dir)

def natural_sort_key(filepath):
    """
    排序规则：
    1. 'Asymptomatic' 强制排在最前 (-1)
    2. 其他文件按文件名中的第一个数字大小排序 (1, 5, 9, 14...)
    """
    filename = os.path.basename(filepath)
    if "Asymptomatic" in filename:
        return -1
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 999

def get_file_line_count(filepath):
    """
    快速计算文件总行数，用于定义进度条的总长度 (Total)
    使用二进制读取模式 ('rb') 以获得最快速度
    """
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f)

def load_dataset_pairs(dataset_name):
    """
    加载指定数据集下的 matrix (.txt) 和 metadata (.csv)
    特性：支持单文件内部进度条显示 (分块读取)
    """
    root_dir = get_project_root() # 获取 HGNA 根目录
    
    # --- 1. 智能路径匹配 (修改版) ---
    # 我们增加了对 "datasets" 文件夹的搜索
    possible_paths = [
        # 优先级 1: HGNA/datasets/COVID19 (这是你现在的结构)
        os.path.join(root_dir, "datasets", dataset_name),
        
        # 优先级 2: HGNA/datasets/COVID19_data (防止你文件夹名字带_data后缀)
        os.path.join(root_dir, "datasets", f"{dataset_name}_data"),
        
        # 优先级 3: 兼容旧模式 (直接在根目录下找)
        os.path.join(root_dir, dataset_name),
        os.path.join(root_dir, f"{dataset_name}_data")
    ]
    
    # 自动在上面列表中寻找第一个存在的路径
    target_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
    if not target_path:
        # 更新报错信息，提示用户我们去 datasets 找过了
        print(f"❌ 错误: 在 'datasets' 文件夹或根目录下未找到 '{dataset_name}'")
        sys.exit(1)

    # --- 2. 搜索文件并排序 ---
    search_pattern = os.path.join(target_path, "*_scaledata.txt")
    txt_files = glob.glob(search_pattern)
    
    # 应用自定义排序
    txt_files.sort(key=natural_sort_key)

    if not txt_files:
        print(f"❌ 错误: 在 {target_path} 中未找到 *_scaledata.txt 文件")
        sys.exit(1)

    # --- 3. 初始化容器 ---
    expr_list = []
    meta_list = []
    file_names = []

    print(f"📂 准备加载 {len(txt_files)} 个数据集来自: {os.path.basename(target_path)}")
    print("-" * 65)

    # --- 4. 逐个文件处理 ---
    for i, txt_path in enumerate(txt_files):
        base_name = os.path.basename(txt_path)
        display_name = base_name.split('_scaledata')[0] # 提取简短名字用于显示
        
        # 推断对应的 CSV 路径
        csv_name = base_name.replace("_scaledata.txt", "_metadata.csv")
        csv_path = os.path.join(target_path, csv_name)

        print(f"[{i+1}/{len(txt_files)}] 正在读取: {display_name}")

        # === 核心：分块读取 Matrix 以显示进度 ===
        
        # 4.1 预估文件大小 (计算行数)
        print(f"   ↳ 正在扫描文件行数...", end="\r")
        total_lines = get_file_line_count(txt_path)
        
        # 4.2 设定分块大小
        # 你的矩阵是 (3000行 x 100000列)，意味着每次 read_csv 需要处理很宽的数据
        # chunksize=100 表示每次读取 100 个基因（行）
        chunk_size = 100 
        
        chunks = []
        
        # 4.3 读取循环
        # index_col=0 表示第一列是基因名
        # sep=r"\s+" 处理空格或制表符分隔
        try:
            with pd.read_csv(txt_path, sep=r"\s+", index_col=0, chunksize=chunk_size) as reader:
                # total_lines - 1 是因为 header 占据了一行，但这只是估算，不减也行
                with tqdm(total=total_lines, unit="row", desc="   ↳ 进度", ncols=80, leave=True) as pbar:
                    for chunk in reader:
                        chunks.append(chunk)
                        pbar.update(len(chunk)) # 更新进度条
            
            # 4.4 合并块
            expr = pd.concat(chunks)
            
        except Exception as e:
            print(f"\n❌ 读取失败: {e}")
            sys.exit(1)

        # === 读取 Metadata ===
        if os.path.exists(csv_path):
            try:
                meta = pd.read_csv(csv_path, index_col=0)
            except Exception as e:
                print(f"   ⚠️  读取 CSV 出错: {e}")
                meta = None
        else:
            print(f"   ⚠️  警告: 缺失 Metadata CSV")
            meta = None
        
        # 存入列表
        expr_list.append(expr)
        meta_list.append(meta)
        file_names.append(base_name)
        
        # 打印完成信息
        print(f"   ✅ 完成. Matrix Shape: {expr.shape}\n")

    return expr_list, meta_list, file_names

if __name__ == "__main__":
    # --- 命令行入口 ---
    parser = argparse.ArgumentParser(description="单细胞数据加载器 (带详细进度条)")
    parser.add_argument('-e', '--experiment', type=str, required=True, help='数据集名称 (例如: COVID19)')
    
    args = parser.parse_args()
    
    # 调用函数
    exprs, metas, names = load_dataset_pairs(args.experiment)
    
    print("🎉 所有数据加载完毕！")