# src/analyze_dataset_distribution.py
"""
Analyze value distribution for all dataset txt files under datasets/<DATASET>.
Reads *_scaledata.txt files via load_dataset_pairs and prints stats.
"""

import argparse
import numpy as np
import pandas as pd

from datasets_loader_bar import load_dataset_pairs


def analyze_expr_distribution(expr, thresholds=None, quantiles=None, name=None):
    title = "Expression Distribution"
    if name:
        title += f" - {name}"
    print(f"\n=== {title} ===")

    arr = expr.to_numpy()
    total = arr.size
    if total == 0:
        print("❌ 空矩阵，无法分析")
        return

    nonzero = arr[arr != 0]
    nnz = nonzero.size
    zero_frac = 1.0 - (nnz / total)
    print(f"Matrix shape: {expr.shape}")
    print(f"Non-zero values: {nnz} / {total} ({(nnz/total):.2%})")
    print(f"Zero fraction  : {zero_frac:.2%}")

    if nnz == 0:
        print("❌ 全零矩阵，无法分析分布")
        return

    s = pd.Series(nonzero)
    print("Non-zero expression value stats:")
    print(s.describe())

    if quantiles is None:
        quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    q_vals = np.quantile(nonzero, quantiles)
    print("\nNon-zero value quantiles:")
    for q, v in zip(quantiles, q_vals):
        print(f"  q{int(q*100):02d}: {v:.6g}")

    if thresholds:
        print("\nThreshold coverage:")
        mask_base = arr != 0
        for t in thresholds:
            above = (arr >= t) & mask_base
            nnz_above = int(above.sum())
            frac = nnz_above / nnz
            print(f"  t={t}: nnz>=t {nnz_above} ({frac:.2%} of non-zero)")

            per_cell = above.sum(axis=0)
            per_gene = above.sum(axis=1)
            print(f"    per-cell (>=t) count stats: {pd.Series(per_cell).describe()}")
            print(f"    per-gene (>=t) count stats: {pd.Series(per_gene).describe()}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze value distribution of all dataset txt files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称（datasets/ 下的文件夹名）"
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="",
        help="表达阈值列表（逗号分隔），如 0.1,0.5,1,2"
    )

    args = parser.parse_args()

    thresholds = []
    if args.thresholds.strip():
        thresholds = [
            float(x) for x in args.thresholds.split(",")
            if x.strip()
        ]

    exprs, metas, names = load_dataset_pairs(args.dataset)
    for expr, name in zip(exprs, names):
        analyze_expr_distribution(
            expr,
            thresholds=thresholds,
            name=name
        )

    print("\n🎉 Dataset distribution analysis 完成")


if __name__ == "__main__":
    main()
