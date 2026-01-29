import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def build_knn_graph(X, k=10):
    """
    X: (N, d)
    return: adjacency matrix A (N, N)
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    _, indices = nbrs.kneighbors(X)

    N = X.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        for j in indices[i]:
            A[i, j] = 1.0
    return A


def build_hypergraph(expr, threshold=0.0):
    """
    expr: (num_cells, num_genes)
    return: incidence matrix H (cells × genes)
    """
    H = (expr > threshold).astype(float)
    return H


def hypergraph_clustering(H, num_clusters):
    """
    Simple hypergraph clustering via k-means on H
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(H)
    centers = kmeans.cluster_centers_
    return labels, centers


def kmeans_align(Z, num_clusters):
    """
    Z: concatenated embeddings
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(Z)
    centers = kmeans.cluster_centers_
    return labels, centers