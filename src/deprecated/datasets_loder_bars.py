import argparse
import os
import glob
import pandas as pd
import sys
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_project_root():
    """获取项目根目录"""
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    return os.path.dirname(src_dir)

def natural_sort_key(filepath):
    """排序规则：Asymptomatic 最前，其余按数字大小"""
    filename = os.path.basename(filepath)
    if "Asymptomatic" in filename:
        return -1
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 999

# --- 核心修改：将单个文件的读取逻辑剥离成一个独立函数 ---
# 这个函数必须放在顶层，以便多进程调用
def process_single_pair(txt_path):
    """
    工作函数：读取一对文件 (txt + csv)
    """
    base_name = os.path.basename(txt_path)
    
    # 推断 CSV 路径
    csv_name = base_name.replace("_scaledata.txt", "_metadata.csv")
    dir_name = os.path.dirname(txt_path)
    csv_path = os.path.join(dir_name, csv_name)
    
    # 1. 读取矩阵 (TXT)
    # 注意：在多进程模式下，我们通常不显示内部的 chunk 进度条，
    # 因为5个进度条混在一起会打乱屏幕显示。
    try:
        expr = pd.read_csv(txt_path, sep=r"\s+", index_col=0)
    except Exception as e:
        return None, None, base_name, f"Error reading txt: {str(e)}"

    # 2. 读取元数据 (CSV)
    if os.path.exists(csv_path):
        try:
            meta = pd.read_csv(csv_path, index_col=0)
        except Exception as e:
            meta = None
    else:
        meta = None
        
    return expr, meta, base_name, None # 最后一个 None 代表无错误

def load_dataset_multicore(dataset_name, max_workers=4):
    """
    多核加载主函数
    """
    root_dir = get_project_root()
    
    # 1. 路径寻找
    possible_paths = [
        os.path.join(root_dir, dataset_name),
        os.path.join(root_dir, f"{dataset_name}_data")
    ]
    target_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
    if not target_path:
        print(f"❌ 错误: 未找到目录 '{dataset_name}'")
        sys.exit(1)

    # 2. 搜索并排序文件
    search_pattern = os.path.join(target_path, "*_scaledata.txt")
    txt_files = glob.glob(search_pattern)
    txt_files.sort(key=natural_sort_key)

    if not txt_files:
        print(f"❌ 错误: 未找到数据文件")
        sys.exit(1)

    # 3. 准备多进程
    # 如果文件数少于核数，就没必要开那么多核
    actual_workers = min(len(txt_files), max_workers)
    
    print(f"🚀 启动多核加速: 使用 {actual_workers} 个核心并行加载 {len(txt_files)} 个文件...")
    print(f"📂 数据源: {os.path.basename(target_path)}")
    print("-" * 60)

    # 4. 并行执行
    results = []
    # ProcessPoolExecutor 自动管理进程池
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        # executor.map 会按照 txt_files 的输入顺序返回结果，这非常重要！
        # 这样我们就不需要重新排序了，只要输入是排好序的，输出就是排好序的。
        # tqdm 用于显示“完成了几个文件”
        results_generator = list(tqdm(
            executor.map(process_single_pair, txt_files), 
            total=len(txt_files), 
            unit="file",
            desc="总进度"
        ))

    # 5. 解包结果
    expr_list = []
    meta_list = []
    file_names = []

    for expr, meta, name, error in results_generator:
        if error:
            print(f"❌ 读取 {name} 失败: {error}")
            sys.exit(1)
            
        expr_list.append(expr)
        meta_list.append(meta)
        file_names.append(name)
        
        # 简单打印每个文件的维度
        meta_shape = meta.shape if meta is not None else "Missing"
        tqdm.write(f"   ✅ {name}: Matrix={expr.shape}, Meta={meta_shape}")

    print("-" * 60)
    return expr_list, meta_list, file_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多核数据加载器")
    parser.add_argument('-e', '--experiment', type=str, required=True, help='数据集名称')
    
    # 新增参数：-j 或 --jobs 指定核数
    parser.add_argument('-j', '--jobs', type=int, default=4, help='使用的CPU核心数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 这里的 names 只是为了让你看下效果，实际返回的就是三个列表
    exprs, metas, names = load_dataset_multicore(args.experiment, max_workers=args.jobs)