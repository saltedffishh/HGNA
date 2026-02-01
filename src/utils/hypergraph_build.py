# src/utils/hypergraph_build.py

import argparse
import numpy as np
import scipy.sparse as sp
import yaml
from pathlib import Path
from sklearn.decomposition import FastICA

from utils.paths import get_experiments_root
from utils.io import load_expr_matrix, save_sparse_matrix


# -------------------------------------------------
# 主流程：构建超图（operator 版本）
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="构建单细胞超图（ICA-based, operator version）"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="experiments/ 下的实验目录名"
    )
    parser.add_argument(
        "--n_programs",
        type=int,
        default=30,
        help="ICA 表达程序数量"
    )
    parser.add_argument(
        "--ica_max_iter",
        type=int,
        default=1000,
        help="ICA 最大迭代次数"
    )
    parser.add_argument(
        "--abs_weight",
        action="store_true",
        help="是否对 ICA cell-program 权重取绝对值（推荐）"
    )
    parser.add_argument(
        "--keep_percentile",
        type=float,
        default=75.0,
        help="构建 H 时保留 Z 的百分位（稀疏化用）"
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # 实验目录
    # -------------------------------------------------
    exp_dir = get_experiments_root() / args.exp_name
    assert exp_dir.exists(), f"❌ 实验目录不存在: {exp_dir}"
    print(f"📂 使用实验目录: {exp_dir}")

    data_dir = exp_dir / "data"
    hyper_dir = exp_dir / "hypergraphs"
    hyper_dir.mkdir(exist_ok=True)

    # 保存超图配置（复现用）
    hyper_cfg = {
        "method": "ICA",
        "n_programs": args.n_programs,
        "ica_max_iter": args.ica_max_iter,
        "abs_weight": args.abs_weight,
        "keep_percentile": args.keep_percentile,
        "laplacian": "implicit_operator"
    }
    with open(hyper_dir / "config.yaml", "w") as f:
        yaml.dump(hyper_cfg, f)

    # -------------------------------------------------
    # 找到所有 stage
    # -------------------------------------------------
    stage_files = sorted(
        [f for f in data_dir.iterdir() if f.name.startswith("expr_stage")]
    )
    print(f"🔍 发现 {len(stage_files)} 个 stage")

    # -------------------------------------------------
    # 逐 stage 构建超图
    # -------------------------------------------------
    for stage_id, expr_file in enumerate(stage_files):
        print(f"\n🧬 Stage {stage_id}: 构建超图")

        # ---------- 1. 读取 scale 表达矩阵 ----------
        X, cells, genes = load_expr_matrix(expr_file)
        # X: (n_cells, n_genes)
        n_cells = X.shape[0]
        print(f"   Expression matrix: {X.shape}")

        # ---------- 2. ICA 学习表达程序 ----------
        ica = FastICA(
            n_components=args.n_programs,
            max_iter=args.ica_max_iter,
            random_state=0
        )
        Z = ica.fit_transform(X)   # (cells × programs)

        if args.abs_weight:
            Z = np.abs(Z)

        # ---------- 3. 稀疏化，构建 H ----------
        thresh = np.percentile(Z, args.keep_percentile)
        Z[Z < thresh] = 0.0

        H = sp.csr_matrix(Z)
        print(f"   Hypergraph H: shape={H.shape}, nnz={H.nnz}")

        # ---------- 4. 计算并保存度信息 ----------
        # 节点度（cells）
        dv = np.array(H.sum(axis=1)).flatten()
        dv[dv == 0] = 1.0  # 防止除零

        # 超边度（programs）
        de = np.array(H.sum(axis=0)).flatten()
        de[de == 0] = 1.0

        print(
            f"   dv: min={dv.min():.2f}, mean={dv.mean():.2f}, max={dv.max():.2f}"
        )
        print(
            f"   de: min={de.min():.2f}, mean={de.mean():.2f}, max={de.max():.2f}"
        )

        # ---------- 5. 缓存 ----------
        save_sparse_matrix(
            hyper_dir / f"H_stage{stage_id}.npz",
            H
        )
        np.save(hyper_dir / f"dv_stage{stage_id}.npy", dv)
        np.save(hyper_dir / f"de_stage{stage_id}.npy", de)

    print("\n✅ 所有 stage 的超图（operator 版本）构建完成")


if __name__ == "__main__":
    main()
