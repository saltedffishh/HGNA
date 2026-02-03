# src/run_experiment.py

# ==================================================
# 0. Path & environment setup (MUST be first)
# ==================================================
import os
import sys
from pathlib import Path

# 项目根目录 = src 的上一层
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================================================
# 1. Standard imports
# ==================================================
import argparse
import numpy as np
import pandas as pd

# ==================================================
# 2. Project imports (REAL interfaces only)
# ==================================================
from datasets_loader_bar import load_dataset_pairs

from hypergraph_functional_pseudocell import (
    functional_pseudocells_from_programs
)

from knn_geometric_refine import refine_with_geometry
from pseudocells import summarize_pseudocells
from UOT import unbalanced_ot_align


# ==================================================
# 3. Experiment pipeline
# ==================================================
def run_experiment(
    dataset,
    exp_name,
    n_programs=30,
    func_k=15,
    func_resolution=1.0,
):
    """
    Full HGNA experiment pipeline

    Stage A: Program-induced functional pseudo-cells
    Stage B: Geometry refinement
    Stage C: Pseudo-cell summarization
    Stage D: Unbalanced OT alignment
    """

    # ---------- experiment output dir ----------
    exp_dir = Path(PROJECT_ROOT) / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Experiment dir: {exp_dir}")

    # ---------- 1. Load dataset ----------
    exprs, metas, names = load_dataset_pairs(dataset)

    prev_summary = None
    align_logs = []

    # ---------- 2. Iterate over stages ----------
    for t, (expr, meta, stage_name) in enumerate(zip(exprs, metas, names)):
        print(f"\n🧪 Stage {t}: {stage_name}")

        # genes × cells → cells × genes
        X = expr.T.values
        print(f"   X shape: {X.shape}")

        # ==================================================
        # Stage A: functional pseudo-cells from programs
        # ==================================================
        # NOTE:
        # - 不显式构建 program-response 矩阵
        # - program / hypergraph 逻辑封装在函数内部
        func_labels = functional_pseudocells_from_programs(
            X,
            n_programs=n_programs,
            k=func_k,
            resolution=func_resolution,
        )

        np.save(exp_dir / f"func_labels_stage{t}.npy", func_labels)
        print(f"   #functional groups: {len(np.unique(func_labels))}")

        # ==================================================
        # Stage B: geometry refinement
        # ==================================================
        refined_labels = refine_with_geometry(
            X,
            func_labels
        )

        np.save(exp_dir / f"labels_stage{t}.npy", refined_labels)
        print(f"   #final pseudo-cells: {len(np.unique(refined_labels))}")

        # ==================================================
        # Stage C: summarize pseudo-cells
        # ==================================================
        summary = summarize_pseudocells(
            X=X,
            labels=refined_labels
        )

        summary.to_csv(
            exp_dir / f"pseudocells_stage{t}.csv",
            index=False
        )

        # ==================================================
        # Stage D: UOT alignment
        # ==================================================
        if prev_summary is not None:
            print("🔗 UOT alignment")
            mu_t = np.vstack(prev_summary["mu"].values)
            mu_t1 = np.vstack(summary["mu"].values)
            pi = unbalanced_ot_align(
                    mu_t,
                    mu_t1,
                    prev_summary["program"].values,
                    summary["program"].values,
                    prev_summary["size"].values,
                    summary["size"].values,
                )
            np.save(exp_dir / f"uot_{t-1}_to_{t}.npy", pi)

            align_logs.append({
                "from": t - 1,
                "to": t,
                "n_src": pi.shape[0],
                "n_tgt": pi.shape[1],
                "mass": float(pi.sum()),
            })

        prev_summary = summary

    # ---------- Save alignment summary ----------
    if align_logs:
        pd.DataFrame(align_logs).to_csv(
            exp_dir / "alignment_summary.csv",
            index=False
        )

    print("\n🎉 Experiment finished successfully")


# ==================================================
# 4. CLI
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HGNA experiment runner"
    )

    # -------- required --------
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g. COVID19"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (output folder)"
    )

    # -------- hyperparameters --------
    parser.add_argument(
        "--n_programs",
        type=int,
        default=30,
        help="Number of expression programs"
    )
    parser.add_argument(
        "--func_k",
        type=int,
        default=15,
        help="KNN k in program space"
    )
    parser.add_argument(
        "--func_resolution",
        type=float,
        default=1.0,
        help="Leiden resolution for functional pseudo-cells"
    )

    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        exp_name=args.exp_name,
        n_programs=args.n_programs,
        func_k=args.func_k,
        func_resolution=args.func_resolution,
    )