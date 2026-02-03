# src/hypergraph_functional_pseudocell.py

import numpy as np
import scanpy as sc


def functional_pseudocells_from_programs(
    X,                 # np.ndarray, shape (n_cells, n_genes)
    n_programs=30,     # number of expression programs (latent dims)
    k=15,              # kNN neighbors in program space
    resolution=1.0,    # Leiden resolution
    pca_seed=0,
):
    """
    Learn expression programs from X and cluster cells in program space.

    Parameters
    ----------
    X : np.ndarray
        Scaled expression matrix (cells × genes)
    n_programs : int
        Number of latent expression programs (PCA dims)
    k : int
        Number of neighbors for kNN graph
    resolution : float
        Leiden resolution parameter
    pca_seed : int
        Random seed for PCA (reproducibility)

    Returns
    -------
    func_labels : np.ndarray
        Functional pseudo-cell labels, shape (n_cells,)
    """

    # -------------------------------
    # 1. Build AnnData from X
    # -------------------------------
    adata = sc.AnnData(X.copy())

    # -------------------------------
    # 2. Learn expression programs
    #    (PCA as linear program basis)
    # -------------------------------
    sc.pp.pca(
        adata,
        n_comps=n_programs,
        svd_solver="arpack",
        random_state=pca_seed,
    )

    # adata.obsm["X_pca"] : (n_cells, n_programs)

    # -------------------------------
    # 3. kNN graph in program space
    # -------------------------------
    sc.pp.neighbors(
        adata,
        n_neighbors=k,
        use_rep="X_pca",
    )

    # -------------------------------
    # 4. Leiden clustering
    # -------------------------------
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added="func_label",
    )

    return adata.obs["func_label"].astype(int).values