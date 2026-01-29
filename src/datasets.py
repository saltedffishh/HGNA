import numpy as np


def generate_cell_data(num_cells=300, num_genes=50, time=0):
    """
    Simulate gene expression with temporal drift
    """
    np.random.seed(42 + time)
    base = np.random.randn(num_cells, num_genes)

    drift = time * 0.3
    expr = base + drift * np.random.randn(num_cells, num_genes)

    return expr