# src/pseudocells.py

import numpy as np
import pandas as pd


def summarize_pseudocells(
    X,          # scale matrix (cells × genes)
    labels,     # pseudo-cell labels (cells,)
):
    """
    Summarize pseudo-cells into a table for alignment.

    Returns a DataFrame with columns:
        - pseudocell_id
        - size
        - mu       : mean expression (flattened)
        - program  : optional low-dim embedding (here: same as mu)
    """

    pseudocell_ids = np.unique(labels)

    records = []

    for pid in pseudocell_ids:
        idx = np.where(labels == pid)[0]
        X_pc = X[idx]

        size = len(idx)
        mu = X_pc.mean(axis=0)   # mean expression (gene space)

        records.append({
            "pseudocell_id": int(pid),
            "size": size,
            # 注意：这里用 list / np.array 存，方便后续 OT
            "mu": mu,
            # program embedding（当前直接用 mu，占位但语义正确）
            "program": mu,
        })

    df = pd.DataFrame(records)

    return df