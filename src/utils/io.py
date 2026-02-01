import numpy as np
import scipy.sparse as sp

def save_expr_matrix(path, X, cells, genes):
    np.savez_compressed(
        path,
        X=X,
        cells=np.array(cells),
        genes=np.array(genes)
    )

def load_expr_matrix(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["cells"].tolist(), data["genes"].tolist()

def save_sparse_matrix(path, M):
    sp.save_npz(path, M)

def load_sparse_matrix(path):
    return sp.load_npz(path)
