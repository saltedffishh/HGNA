import pandas as pd
import numpy as np

def inspect_scaled_dataset_for_hypergraph(file_path):
    print(f"正在读取文件: {file_path} ...")
    
    try:
        # 1. 读取数据
        # 根据截图，数据是空格分隔，且包含引号
        # index_col=0 将第一列(基因名)作为行索引
        df = pd.read_csv(file_path, sep=r'\s+', index_col=0, quotechar='"')
        
        # 转换为 numpy 矩阵以提高计算速度
        matrix = df.values
        
        print("\n=== 1. 数据维度检查 ===")
        print(f"• 基因数 (Rows/Nodes?): {df.shape[0]}")
        print(f"• 细胞数 (Cols/Hyperedges?): {df.shape[1]}")
        print(f"• 示例基因: {list(df.index[:3])}")
        print(f"• 示例细胞: {list(df.columns[:3])}")
        
        print("\n=== 2. 数值分布诊断 (确认是否为 Scaled) ===")
        min_val, max_val = matrix.min(), matrix.max()
        mean_val, std_val = matrix.mean(), matrix.std()
        
        print(f"• 最小值 (Min): {min_val:.4f}")
        print(f"• 最大值 (Max): {max_val:.4f}")
        print(f"• 均值 (Mean):   {mean_val:.4f} (预期接近 0)")
        print(f"• 标准差 (Std): {std_val:.4f} (预期接近 1)")
        
        if min_val < 0 and abs(mean_val) < 0.1:
            print("✅ 结论: 数据已完成 Z-score Standardization (Scale)。")
            print("⚠️ 警告: 严禁再次进行 Log1p 或 Normalization 操作。")
        else:
            print("❓ 结论: 数据分布异常，可能并非标准的 Scaled 数据。")

        print("\n=== 3. 超图构建可行性测试 (关键) ===")
        print("说明: 由于数据是连续值，您需要设定阈值来定义'节点属于超边'。")
        print("以下测试不同阈值下的网络稀疏度（即保留了多少连接）：")
        
        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]
        total_elements = matrix.size
        
        print(f"{'阈值 (Theta)':<15} | {'保留连接数':<12} | {'稀疏度 (%)':<15} | {'建议'}")
        print("-" * 65)
        
        for theta in thresholds:
            # 计算大于阈值的元素个数
            count = np.sum(matrix > theta)
            sparsity = (count / total_elements) * 100
            
            recommendation = ""
            if 5 < sparsity < 20:
                recommendation = "★ 推荐 (信息量适中)"
            elif sparsity > 50:
                recommendation = "过密 (包含太多噪声)"
            elif sparsity < 0.1:
                recommendation = "过稀 (可能断连)"
                
            print(f"> {theta:<13} | {count:<12} | {sparsity:>6.2f}%        | {recommendation}")

        print("-" * 65)
        print("💡 提示: 对于 HGNA (网络对齐) 任务，建议选择稀疏度在 5%-15% 左右的阈值。")
        print("        如果选择 > 0 (50%保留)，对于超图算法来说通常太稠密了，计算量会爆炸。")

    except Exception as e:
        print(f"❌ 读取错误: {e}")
        print("请检查路径是否正确，或文件格式是否与截图一致。")

# --- 运行部分 ---
# 请修改这里的路径
file_path = '/Users/user/Desktop/任务/HGNA/datasets/COVID19/Asymptomatic_scaledata.txt' 
inspect_scaled_dataset_for_hypergraph(file_path)