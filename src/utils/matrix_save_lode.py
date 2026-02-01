import numpy as np

def save_expr_matrix(path, X, cells, genes):
    """
    X: np.ndarray (cells × genes)
    """
    np.savez_compressed(
        path,
        X=X,
        cells=np.array(cells),
        genes=np.array(genes)
    )

def load_expr_matrix(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["cells"].tolist(), data["genes"].tolist()
