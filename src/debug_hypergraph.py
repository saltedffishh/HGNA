# src/debug_hypergraph.py
"""
Hypergraph sanity check (operator version)

Checks:
1. Structure sanity (H shape / sparsity)
2. Degree statistics (dv / de)
3. Cell participation distribution
4. Program size distribution
5. Biological enrichment using cell.annotation
"""

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.paths import get_experiments_root
from utils.io import load_sparse_matrix
from datasets_loader_bar import load_dataset_pairs
from utils.io import load_expr_matrix


# -------------------------------------------------
# Check 1: 基本结构
# -------------------------------------------------
def check_structure(H):
    print("\n=== Check 1: Hypergraph Structure ===")
    print(f"H shape      : {H.shape}")
    print(f"H nnz        : {H.nnz}")
    print(f"H is sparse  : {sp.issparse(H)}")

    assert sp.issparse(H), "❌ H 必须是稀疏矩阵"
    assert H.shape[0] > H.shape[1], "❌ modules 数异常大？"

    print("✅ Check 1 passed")


# -------------------------------------------------
# Check 2: 度分布
# -------------------------------------------------
def check_degrees(dv, de):
    print("\n=== Check 2: Degree Distributions ===")

    print("Cell degree (dv):")
    print(pd.Series(dv).describe())

    print("\nProgram degree (de):")
    print(pd.Series(de).describe())

    print("Zero dv count:", np.sum(dv == 0))
    print("Zero de count:", np.sum(de == 0))

    print("✅ Check 2 finished (需人工判断是否合理)")


# -------------------------------------------------
# Check 3: cell 参与 program 数
# -------------------------------------------------
def check_cell_participation(H):
    print("\n=== Check 3: Cell Participation ===")

    counts = np.array((H > 0).sum(axis=1)).flatten()
    stats = pd.Series(counts).describe()
    print(stats)

    print(
        f"Cells with zero programs: {np.sum(counts == 0)}"
    )

    print("✅ Check 3 finished")


# -------------------------------------------------
# Check 4: program 覆盖细胞数
# -------------------------------------------------
def check_program_sizes(H):
    print("\n=== Check 4: Program Sizes ===")

    sizes = np.array((H > 0).sum(axis=0)).flatten()
    stats = pd.Series(sizes).describe()
    print(stats)

    print("Smallest programs:", np.sort(sizes)[:5])
    print("Largest programs :", np.sort(sizes)[-5:])

    print("✅ Check 4 finished")


# -------------------------------------------------
# Check 5: 生物学一致性（program 富集）
# -------------------------------------------------
def check_biological_enrichment(H, meta, top_k=3):
    print("\n=== Check 5: Biological Enrichment ===")

    assert "cell.annotation" in meta.columns, \
        "❌ metadata 中缺少 cell.annotation"

    H_bin = (H > 0).astype(int)

    for m in range(min(H.shape[1], top_k)):
        cells_in_prog = np.where(H_bin[:, m].toarray().flatten())[0]
        if len(cells_in_prog) == 0:
            continue

        print(f"\nProgram {m}: size = {len(cells_in_prog)}")
        print(
            meta.iloc[cells_in_prog]["cell.annotation"]
            .value_counts(normalize=True)
            .head()
        )

    print("✅ Check 5 finished")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hypergraph sanity check (operator version)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称（用于 metadata）"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="experiments/ 下实验目录名"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="检查第几个 stage"
    )

    args = parser.parse_args()

    # -----------------------------
    # 实验目录
    # -----------------------------
    exp_dir = get_experiments_root() / args.exp_name
    assert exp_dir.exists(), f"❌ 实验目录不存在: {exp_dir}"
    print(f"📂 使用实验目录: {exp_dir}")

    hyper_dir = exp_dir / "hypergraphs"

    # -----------------------------
    # 读取超图
    # -----------------------------
    H = load_sparse_matrix(
        hyper_dir / f"H_stage{args.stage}.npz"
    )
    dv = np.load(hyper_dir / f"dv_stage{args.stage}.npy")
    de = np.load(hyper_dir / f"de_stage{args.stage}.npy")

    # -----------------------------
    # 读取 metadata
    # -----------------------------
    exprs, metas, names = load_dataset_pairs(args.dataset)
    # 读取 cell_id（和 H 行顺序一致）
    X, cells, genes = load_expr_matrix(
        exp_dir / "data" / f"expr_stage{args.stage}.npz"
    )

    # 用 cell_id 对齐 metadata（这是关键）
    meta = metas[args.stage].loc[cells]


    # -----------------------------
    # 执行检查
    # -----------------------------
    check_structure(H)
    check_degrees(dv, de)
    check_cell_participation(H)
    check_program_sizes(H)
    check_biological_enrichment(H, meta)

    print("\n🎉 Hypergraph sanity check 完成")


if __name__ == "__main__":
    main()
