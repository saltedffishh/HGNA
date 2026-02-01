# src/build_knn_graph.py

import argparse
import yaml

from datasets_loder_bar import load_dataset_pairs
from utils import knn_build
from utils.paths import (
    create_experiment_dir,
    get_expr_cache_path,
    get_knn_cache_path,
    get_config_path
)
from utils.io import (
    save_expr_matrix,
    load_expr_matrix,
    save_sparse_matrix,
    load_sparse_matrix
)

def main():
    # -----------------------------
    # 1. 命令行参数
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="构建单细胞 KNN 图（实验级缓存）"
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        required=True,
        help="数据集名称，例如 COVID19"
    )
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--pca_dim", type=int, default=50)

    args = parser.parse_args()

    # -----------------------------
    # 2. 实验配置
    # -----------------------------
    config = {
        "k": args.k,
        "pca": args.pca_dim,
        "experiment": args.experiment
    }

    # ⚠️ main 不关心 experiments 在哪
    exp_dir = create_experiment_dir(config)
    print(f"🧪 创建实验目录: {exp_dir}")

    # 保存配置
    with open(get_config_path(exp_dir), "w") as f:
        yaml.dump(config, f)

    # -----------------------------
    # 3. 读取原始数据（只一次）
    # -----------------------------
    exprs, metas, names = load_dataset_pairs(args.experiment)

    # -----------------------------
    # 4. 逐 stage 处理
    # -----------------------------
    for idx, (expr, meta, name) in enumerate(zip(exprs, metas, names)):
        print(f"\n🔗 Stage {idx}: {name}")

        # ---------- 4.1 表达矩阵缓存 ----------
        expr_cache = get_expr_cache_path(exp_dir, idx)

        if expr_cache.exists():
            print("♻️  使用缓存表达矩阵")
            X, cells, genes = load_expr_matrix(expr_cache)
        else:
            print("📥 保存表达矩阵缓存")
            X = expr.T.values
            cells = expr.columns.tolist()
            genes = expr.index.tolist()
            save_expr_matrix(expr_cache, X, cells, genes)

        # ---------- 4.2 KNN 图缓存 ----------
        knn_cache = get_knn_cache_path(exp_dir, idx)

        if knn_cache.exists():
            print("♻️  使用缓存 KNN 图")
            L_G = load_sparse_matrix(knn_cache)
        else:
            print("🔨 构建新的 KNN 图")
            L_G = knn_build.build_knn_graph_from_expr(
                expr,
                k=args.k,
                n_pca=args.pca_dim
            )
            save_sparse_matrix(knn_cache, L_G)

    print("\n✅ 所有 KNN 图构建并缓存完成")

if __name__ == "__main__":
    main()
