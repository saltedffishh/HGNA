# src/utils/knn_build.py

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA

def expr_df_to_cell_matrix(expr_df, n_pca=50):
    X = expr_df.T.values
    X_pca = PCA(n_components=n_pca, random_state=0).fit_transform(X)
    return X_pca

def build_knn_adjacency(X, k=15):
    A = kneighbors_graph(
        X,
        n_neighbors=k,
        mode="connectivity",
        include_self=False
    )
    return A  # 🔥 保持 sparse

def adjacency_to_laplacian(A):
    """
    A: scipy.sparse matrix
    """
    # 对称化
    A = 0.5 * (A + A.T)

    # 度矩阵（仍然 sparse）
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)

    L = D - A
    return L

def build_knn_graph_from_expr(expr_df, k=15, n_pca=50):
    X = expr_df_to_cell_matrix(expr_df, n_pca)
    A = build_knn_adjacency(X, k)
    L_G = adjacency_to_laplacian(A)
    return L_G
