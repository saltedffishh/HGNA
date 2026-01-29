import numpy as np
from utils import kmeans_align


class TemporalAligner:
    def __init__(self, num_align_clusters):
        self.num_align_clusters = num_align_clusters

    def align(self, pseudocells_by_time):
        """
        pseudocells_by_time: list of (num_pseudo, dim)
        """
        Z = np.concatenate(pseudocells_by_time, axis=0)

        labels, centers = kmeans_align(
            Z, self.num_align_clusters
        )

        aligned = {}
        idx = 0
        for t, pcs in enumerate(pseudocells_by_time):
            aligned[t] = labels[idx:idx + pcs.shape[0]]
            idx += pcs.shape[0]

        return aligned, centers