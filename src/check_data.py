import argparse
import pandas as pd
import numpy as np
import sys

# 引入你刚才写的加载器
# 假设你的加载脚本叫 datasets_loder.py (在同一目录下)
from datasets_loader_bar import load_dataset_pairs

def check_data_quality(expr_list, meta_list, names):
    """
    对加载的数据进行全面体检
    """
    print("\n" + "="*60)
    print("🔬 开始数据质量验证 (Data Quality Check)")
    print("="*60)

    for i, name in enumerate(names):
        print(f"\n📂 数据集 [{i+1}]: {name}")
        
        expr = expr_list[i]
        meta = meta_list[i]
        
        # --- 1. 基础维度检查 ---
        n_genes, n_cells_expr = expr.shape
        print(f"   🔹 矩阵维度: {n_genes} 基因 x {n_cells_expr} 细胞")
        
        if meta is not None:
            n_cells_meta, n_features = meta.shape
            print(f"   🔹 元数据维度: {n_cells_meta} 细胞 x {n_features} 特征")
            
            # --- 2. 关键：细胞对齐检查 (Alignment Check) ---
            # 检查矩阵的列名 (Cell IDs) 是否与元数据的行名一致
            if n_cells_expr != n_cells_meta:
                print(f"   ❌ 严重错误: 细胞数量不匹配! (矩阵: {n_cells_expr} vs 元数据: {n_cells_meta})")
            else:
                # 检查 ID 是否完全一致且顺序相同
                # 这一步非常重要，防止张冠李戴
                if expr.columns.equals(meta.index):
                    print(f"   ✅ 对齐检查: 通过 (矩阵列名与元数据行名完全一致)")
                else:
                    # 如果数量一样但顺序不一样，尝试看是否集合相同
                    if set(expr.columns) == set(meta.index):
                         print(f"   ⚠️ 警告: 细胞ID相同但顺序不同，建议重新排序！")
                    else:
                         print(f"   ❌ 严重错误: 细胞ID不匹配！")

        else:
            print("   ⚠️ 警告: 该数据集没有对应的 Metadata")

        # --- 3. 空值检查 (NaN Check) ---
        if expr.isnull().values.any():
            nan_count = expr.isnull().sum().sum()
            print(f"   ❌ 发现空值: 矩阵中共有 {nan_count} 个 NaN")
        else:
            print(f"   ✅ 完整性检查: 无空值 (No NaN)")

        # --- 4. 数据值抽样 (Value Check) ---
        # 看看前3行前3列，确认读进来的是数字
        print(f"   👀 数据预览 (Top-left 3x3):")
        print(expr.iloc[:3, :3].to_string())

    print("\n" + "="*60)
    print("🎉 验证结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, required=True, help='数据集名称')
    args = parser.parse_args()

    # 1. 调用你之前的加载函数
    print("正在加载数据以进行检查...")
    # 注意：这里调用的是你刚才写好的 datasets_loder 里的函数
    exprs, metas, names = load_dataset_pairs(args.experiment)
    
    # 2. 执行检查
    check_data_quality(exprs, metas, names)