# src/debug_knn.py
"""
KNN 图完整 sanity check 脚本（研究级版本）

覆盖：
Check 1: 结构正确性
Check 2: 度分布
Check 3: 几何合理性（邻居 vs 非邻居距离）
Check 4: 生物学一致性（cell type 富集）

⚠️ 重要设计原则：
- experiments 只存数值缓存（expr / graph）
- dataset 提供语义（metadata）
"""
#运行 python -m debug_knn   --dataset COVID19   --exp_name 2026-01-31_15-20-18__k15_pca50_experimentCOVID19   --stage 0   --k 15

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.io import load_expr_matrix, load_sparse_matrix
from utils.paths import get_experiments_root
from datasets_loader_bar import load_dataset_pairs



# -------------------------------------------------
# Check 1: 结构检查
# -------------------------------------------------
def check_structure(X, L_G, k):
    print("\n=== Check 1: Graph Structure ===")

    print(f"Expression shape : {X.shape}")
    print(f"Laplacian shape  : {L_G.shape}")

    assert X.shape[0] == L_G.shape[0], "❌ 细胞数不一致"
    assert L_G.shape[0] == L_G.shape[1], "❌ Laplacian 不是方阵"

    print("Is sparse        :", sp.issparse(L_G))
    assert sp.issparse(L_G), "❌ L_G 不是稀疏矩阵"

    nnz = L_G.nnz
    expected = X.shape[0] * k * 2
    print(f"nnz              : {nnz}")
    print(f"expected (~)     : {expected}")

    print("✅ Check 1 passed")


# -------------------------------------------------
# Check 2: 度分布
# -------------------------------------------------
def check_degree_distribution(L_G, k):
    print("\n=== Check 2: Degree Distribution ===")

    degrees = np.array(L_G.diagonal())
    stats = pd.Series(degrees).describe()
    print(stats)

    num_zero = np.sum(degrees == 0)
    print(f"Zero-degree nodes: {num_zero}")

    if num_zero > 0:
        print("⚠️  存在孤立节点（需检查 KNN 构建）")
    else:
        print("✅ 无孤立节点")

    print(f"Expected range: min ≈ {k}, max ≈ {2 * k}")
    print("✅ Check 2 finished（需人工判断是否合理）")


# -------------------------------------------------
# Check 3: 几何合理性
# -------------------------------------------------
def check_geometric_validity(X, L_G, num_trials=5):
    print("\n=== Check 3: Geometric Validity ===")

    n = X.shape[0]

    for t in range(num_trials):
        i = np.random.randint(0, n)
        neighbors = L_G[i].nonzero()[1]
        neighbors = neighbors[neighbors != i]

        if len(neighbors) == 0:
            print(f"[Trial {t}] Cell {i}: no neighbors, skip")
            continue

        dist_neighbors = np.linalg.norm(X[neighbors] - X[i], axis=1)

        non_neighbors = np.setdiff1d(np.arange(n), neighbors)
        sample = np.random.choice(non_neighbors, size=len(neighbors), replace=False)
        dist_non = np.linalg.norm(X[sample] - X[i], axis=1)

        print(
            f"[Trial {t}] "
            f"neighbor mean = {dist_neighbors.mean():.4f}, "
            f"non-neighbor mean = {dist_non.mean():.4f}"
        )

    print("✅ Check 3 finished（邻居距离应显著更小）")


# -------------------------------------------------
# Check 4: 生物学一致性
# -------------------------------------------------
def check_biological_validity(L_G, meta, num_trials=5):
    print("\n=== Check 4: Biological Validity ===")

    assert "cell.annotation" in meta.columns, \
        "❌ metadata 中缺少 cell.annotation"

    for t in range(num_trials):
        i = np.random.randint(0, len(meta))
        neighbors = L_G[i].nonzero()[1]

        if len(neighbors) == 0:
            print(f"[Trial {t}] Cell {i}: no neighbors, skip")
            continue

        cell_type = meta.iloc[i]["cell.annotation"]
        neighbor_types = meta.iloc[neighbors]["cell.annotation"]

        print(f"\n[Trial {t}] Cell {i} type = {cell_type}")
        print(neighbor_types.value_counts(normalize=True).head())

    print("✅ Check 4 finished（观察是否有 cell-type 富集）")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="KNN 图完整 sanity check（不重新计算）"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称，例如 COVID19（用于加载 metadata）"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="experiments/ 下的实验文件夹名"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="检查第几个 stage（默认 0）"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="KNN 的 k（用于 sanity reference）"
    )

    args = parser.parse_args()

    # ---------- 实验目录 ----------
    exp_dir = get_experiments_root() / args.exp_name
    assert exp_dir.exists(), f"❌ 实验目录不存在: {exp_dir}"
    print(f"📂 使用实验目录: {exp_dir}")

    # ---------- 加载缓存 ----------
    X, cells, genes = load_expr_matrix(
        exp_dir / "data" / f"expr_stage{args.stage}.npz"
    )
    L_G = load_sparse_matrix(
        exp_dir / "graphs" / f"knn_L_stage{args.stage}.npz"
    )

    # ---------- 加载 metadata（语义来源） ----------
    exprs, metas, names = load_dataset_pairs(args.dataset)
    meta = metas[args.stage].loc[cells]

    # ---------- 执行所有检查 ----------
    check_structure(X, L_G, args.k)
    check_degree_distribution(L_G, args.k)
    check_geometric_validity(X, L_G)
    check_biological_validity(L_G, meta)

    print("\n🎉 所有 KNN sanity check 完成")


if __name__ == "__main__":
    main()
