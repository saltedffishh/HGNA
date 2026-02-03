# src/knn_geometric_refine.py

import numpy as np
import scanpy as sc

def refine_with_geometry(
    X,
    func_labels,
    pca_dim=30,
    knn_k=15,
    resolution=0.5,
):
    """
    X: np.ndarray, (n_cells, n_genes), scale matrix
    func_labels: np.ndarray, functional cluster labels

    return:
        refined_labels: np.ndarray, final pseudo-cell labels
    """
    n_cells = X.shape[0]
    refined_labels = -np.ones(n_cells, dtype=int)
    current_label = 0

    for f in np.unique(func_labels):
        idx = np.where(func_labels == f)[0]
        if len(idx) < 30:
            refined_labels[idx] = current_label
            current_label += 1
            continue

        adata = sc.AnnData(X[idx])
        sc.pp.pca(adata, n_comps=pca_dim)
        sc.pp.neighbors(adata, n_neighbors=knn_k)
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added="geo_label"
        )

        sub_labels = adata.obs["geo_label"].astype(int).values
        for sl in np.unique(sub_labels):
            refined_labels[idx[sub_labels == sl]] = current_label
            current_label += 1

    return refined_labels