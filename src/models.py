import numpy as np
from utils import build_hypergraph, hypergraph_clustering


class PseudoCellGenerator:
    def __init__(self, num_pseudocells):
        self.num_pseudocells = num_pseudocells

    def fit(self, expr):
        """
        expr: (cells, genes)
        """
        H = build_hypergraph(expr)
        labels, centers = hypergraph_clustering(
            H, self.num_pseudocells
        )

        self.labels_ = labels
        self.pseudocells_ = centers
        return centers, labels