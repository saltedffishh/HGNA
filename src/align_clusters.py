# src/align_clusters.py
"""
Align clusters between adjacent stages using centroid matching (Hungarian).
Requires per-stage Z and labels saved in experiments/<exp_name>/align/.
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from utils.paths import get_experiments_root


def load_stage_data(align_dir, stage):
    z_path = align_dir / f"stage{stage}_Z.npy"
    labels_path = align_dir / f"stage{stage}_labels.npy"
    if not z_path.exists():
        raise FileNotFoundError(f"缺少 Z: {z_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"缺少 labels: {labels_path}")
    Z = np.load(z_path)
    labels = np.load(labels_path)
    return Z, labels


def compute_centroids(Z, labels, n_clusters):
    centroids = np.zeros((n_clusters, Z.shape[1]), dtype=np.float32)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            centroids[c] = np.nan
        else:
            centroids[c] = Z[idx].mean(axis=0)
    return centroids


def align_labels(Z_a, labels_a, Z_b, labels_b, n_clusters):
    cent_a = compute_centroids(Z_a, labels_a, n_clusters)
    cent_b = compute_centroids(Z_b, labels_b, n_clusters)

    # 距离矩阵 (n_clusters x n_clusters)
    dist = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    for i in range(n_clusters):
        for j in range(n_clusters):
            if np.any(np.isnan(cent_a[i])) or np.any(np.isnan(cent_b[j])):
                dist[i, j] = 1e9
            else:
                diff = cent_a[i] - cent_b[j]
                dist[i, j] = np.sqrt((diff * diff).sum())

    row_ind, col_ind = linear_sum_assignment(dist)
    mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}

    aligned = labels_b.copy()
    for old, new in mapping.items():
        aligned[labels_b == old] = new

    return aligned, mapping, dist


def align_adjacent_stages(exp_dir, stages, n_clusters):
    align_dir = exp_dir / "align"
    if not align_dir.exists():
        raise FileNotFoundError(f"未找到 align 目录: {align_dir}")

    if len(stages) < 2:
        raise ValueError("至少需要两个 stage 才能对齐")

    for i in range(len(stages) - 1):
        a = stages[i]
        b = stages[i + 1]
        Z_a, labels_a = load_stage_data(align_dir, a)
        Z_b, labels_b = load_stage_data(align_dir, b)

        aligned_b, mapping, dist = align_labels(
            Z_a, labels_a, Z_b, labels_b, n_clusters
        )

        out_path = align_dir / f"stage{b}_labels_aligned_to_{a}.npy"
        np.save(out_path, aligned_b)

        map_path = align_dir / f"stage{b}_mapping_to_{a}.txt"
        with open(map_path, "w") as f:
            for old in sorted(mapping.keys()):
                f.write(f"{old} -> {mapping[old]}\n")

        print(f"Aligned stage {b} to {a}")
        print(f"Saved aligned labels: {out_path}")
        print(f"Saved mapping: {map_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Align clusters between adjacent stages"
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--stages", type=str, required=True, help="例如 0,1,2,3")
    parser.add_argument("--n_clusters", type=int, required=True)
    args = parser.parse_args()

    exp_dir = get_experiments_root() / args.exp_name
    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    align_adjacent_stages(exp_dir, stages, args.n_clusters)


if __name__ == "__main__":
    main()
